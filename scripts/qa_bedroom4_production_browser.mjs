#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { chromium } from "@playwright/test";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");
const DEFAULT_WORLD_DIR = path.join(repoRoot, "web/public/worlds/bedroom4");
const DEFAULT_EXAMPLE_DIR = path.join(repoRoot, "examples/bedroom4");

function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const key = token.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for --${key}`);
    options[key] = value;
    index += 1;
  }
  return options;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, payload) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(payload, null, 2)}\n`);
  fs.renameSync(temporary, filePath);
}

function portableReport(value, worldDir) {
  if (Array.isArray(value)) return value.map((item) => portableReport(item, worldDir));
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [
      key,
      portableReport(item, worldDir),
    ]));
  }
  if (typeof value !== "string" || !value.startsWith("/")) return value;
  if (value === worldDir || value.startsWith(`${worldDir}${path.sep}`)) {
    const relative = path.relative(worldDir, value).split(path.sep).join("/");
    return `web-bundle://bedroom4/${relative}`;
  }
  if (value === repoRoot || value.startsWith(`${repoRoot}${path.sep}`)) {
    const relative = path.relative(repoRoot, value).split(path.sep).join("/");
    return `repo://video2world/${relative}`;
  }
  return `artifact://external/${path.basename(value)}`;
}

function sha256File(filePath) {
  const hash = crypto.createHash("sha256");
  hash.update(fs.readFileSync(filePath));
  return hash.digest("hex");
}

function rounded(value, digits = 4) {
  return Number(Number(value).toFixed(digits));
}

function vectorDistance(left, right) {
  return Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z);
}

function relativeVectorError(actual, expected) {
  return Math.max(...["x", "y", "z"].map((axis) => (
    Math.abs(actual[axis] - expected[axis]) / Math.max(1e-6, Math.abs(expected[axis]))
  )));
}

function assert(condition, message, failures) {
  if (!condition) failures.push(message);
}

export function validatePromotionReport({ browserReport, manifest, currentManifestSha256 }) {
  if (browserReport.automatedGate !== "passed" || browserReport.status !== "candidate") {
    throw new Error("Cannot promote: browser report must be a passed candidate gate");
  }
  if (!Array.isArray(browserReport.failures) || browserReport.failures.length !== 0) {
    throw new Error("Cannot promote: browser report contains failures");
  }
  if (browserReport.manifestSha256 !== currentManifestSha256) {
    throw new Error("Cannot promote: browser report does not match the current manifest hash");
  }

  const definitions = Array.isArray(manifest.interactiveObjects) ? manifest.interactiveObjects : [];
  const objectIds = definitions.map((object) => object.id).sort();
  const collidableIds = definitions
    .filter((object) => object.collision?.mode !== "none")
    .map((object) => object.id)
    .sort();
  const reportObjectIds = (browserReport.objects || []).map((object) => object.id).sort();
  const collisionObjectIds = (browserReport.collisionWorld?.objects || []).map((object) => object.id).sort();
  const robotObjectIds = (browserReport.interactions?.robotCollisions || [])
    .filter((item) => item.passed === true)
    .map((item) => item.objectId)
    .sort();
  if (JSON.stringify(reportObjectIds) !== JSON.stringify(objectIds)) {
    throw new Error("Cannot promote: browser report does not cover every visual object");
  }
  if (JSON.stringify(collisionObjectIds) !== JSON.stringify(objectIds)) {
    throw new Error("Cannot promote: collision report does not cover every object");
  }
  if (JSON.stringify(robotObjectIds) !== JSON.stringify(collidableIds)) {
    throw new Error("Cannot promote: robot collision report does not cover every collider");
  }
  if (browserReport.counts?.visualObjectsExpected !== objectIds.length) {
    throw new Error("Cannot promote: visual object count differs from the current manifest");
  }
  if (browserReport.counts?.objectCollidersExpected !== collidableIds.length) {
    throw new Error("Cannot promote: collider count differs from the current manifest");
  }
  return true;
}

function promotePlacementGate({ worldDir, browserReportPath, exampleBrowserReport }) {
  const manifestPath = path.join(worldDir, "manifest.json");
  const bundleReportPath = path.join(worldDir, "qa/production-bundle-report.json");
  const exampleManifestPath = path.join(DEFAULT_EXAMPLE_DIR, "manifest.production.json");
  const exampleBundleReportPath = path.join(DEFAULT_EXAMPLE_DIR, "production-bundle-report.json");
  let browserReport = readJson(browserReportPath);
  const manifest = readJson(manifestPath);
  validatePromotionReport({
    browserReport,
    manifest,
    currentManifestSha256: sha256File(manifestPath),
  });
  for (const object of manifest.interactiveObjects || []) {
    if (!object.collision?.gate) throw new Error(`${object.id}: missing collision gate`);
    if (object.collision.mode === "none") {
      if (object.collision.gate.status !== "not_tested" || object.collision.asset != null) {
        throw new Error(`${object.id}: visual-only collision truth boundary changed`);
      }
      continue;
    }
    object.collision.gate = {
      ...object.collision.gate,
      status: "passed",
      orientationEvidence: "browser_gaussian_collision_overlay_reviewed",
      browserOverlayReview: "passed",
      note: "Automated browser collision QA and screenshot overlay review passed.",
    };
  }
  manifest.productionBuild.objectPlacementGate = "passed_browser_overlay_and_robot_collision";
  manifest.productionBuild.pillowVisualInteractionGate = "passed_browser_points_bbox_drag_spin";
  manifest.productionBuild.browserQaReport = "./worlds/bedroom4/qa/browser-qa.json";
  writeJson(manifestPath, manifest);

  const bundleReport = readJson(bundleReportPath);
  bundleReport.status = "passed";
  bundleReport.manifestSha256 = sha256File(manifestPath);
  bundleReport.gates.objectPlacementBrowserQa = "passed";
  bundleReport.gates.pillowVisualComponent = "passed";
  for (const collision of bundleReport.collisions || []) {
    collision.gate = {
      ...collision.gate,
      status: "passed",
      orientationEvidence: "browser_gaussian_collision_overlay_reviewed",
      browserOverlayReview: "passed",
      note: "Automated browser collision QA and screenshot overlay review passed.",
    };
  }
  writeJson(bundleReportPath, bundleReport);

  browserReport.status = "passed";
  browserReport.placementReview = {
    status: "passed",
    reviewer: "codex_screenshot_review",
    reviewedAt: new Date().toISOString(),
  };
  browserReport.manifestSha256AfterPromotion = sha256File(manifestPath);
  browserReport = portableReport(browserReport, worldDir);
  writeJson(browserReportPath, browserReport);
  writeJson(exampleManifestPath, manifest);
  writeJson(exampleBundleReportPath, bundleReport);
  writeJson(exampleBrowserReport, browserReport);
  console.log(JSON.stringify({ status: "passed", manifestPath, browserReportPath }, null, 2));
}

async function runBrowserQa({ url, worldDir, browserReportPath, exampleBrowserReport, screenshotDir }) {
  const manifestPath = path.join(worldDir, "manifest.json");
  const manifest = readJson(manifestPath);
  const objectDefinitions = manifest.interactiveObjects || [];
  const objectIds = objectDefinitions.map((object) => object.id);
  const collidableObjectIds = objectDefinitions
    .filter((object) => object.collision?.mode !== "none")
    .map((object) => object.id);
  const pillowId = "sam3_pillow_01";
  if (objectIds.length !== 5 || !objectIds.includes(pillowId)) {
    throw new Error(`Expected five visual objects including ${pillowId}, found ${objectIds.join(",")}`);
  }
  if (collidableObjectIds.length !== 4 || collidableObjectIds.includes(pillowId)) {
    throw new Error(`Expected four non-pillow object colliders, found ${collidableObjectIds.join(",")}`);
  }
  fs.mkdirSync(screenshotDir, { recursive: true });

  const consoleMessages = [];
  const pageErrors = [];
  const requestFailures = [];
  const failures = [];
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
  });
  const page = await context.newPage();
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      consoleMessages.push({ type: message.type(), text: message.text() });
    }
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("requestfailed", (request) => {
    requestFailures.push({ url: request.url(), error: request.failure()?.errorText || "unknown" });
  });

  const startedAt = new Date().toISOString();
  const startedMs = Date.now();
  let report;
  try {
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
    await page.waitForFunction(() => {
      const state = window.__visualPhysicsDemoState;
      return state?.visualReady
        && state?.colliderReady
        && state?.sceneQaReady
        && state?.interactiveObjectReadyCount === 5
        && state?.objectColliderReadyCount === 4;
    }, null, { timeout: 300_000 });
    await page.evaluate(() => window.__setObjectColliderDebugVisible(true));
    await page.waitForTimeout(1_500);

    const initialState = await page.evaluate(() => window.__visualPhysicsDemoState);
    const collisionWorld = await page.evaluate(() => window.__inspectCollisionWorld());
    const pillowCarve = manifest.assets.visual?.pillowCarve;
    assert(pillowCarve?.status === "passed", "pillow static-visual carve gate is not passed", failures);
    assert(
      pillowCarve?.method === "accepted_mask_projected_point_nearest_neighbor_within_exact_aabb",
      `pillow carve method=${pillowCarve?.method}`,
      failures,
    );
    assert(pillowCarve?.distance === 0.08, `pillow carve distance=${pillowCarve?.distance}`, failures);
    assert(
      pillowCarve?.removedGaussianCount === 26_731
        && pillowCarve?.retainedInsideAcceptedAabbCount === 4_038
        && pillowCarve?.removedOutsideAcceptedAabbCount === 0,
      `pillow carve counts=${JSON.stringify(pillowCarve)}`,
      failures,
    );
    assert(initialState.staticVisualCount === 823_391, `staticVisualCount=${initialState.staticVisualCount}`, failures);
    assert(
      initialState.interactiveObjectGaussianCount === 480_000,
      `interactiveObjectGaussianCount=${initialState.interactiveObjectGaussianCount}`,
      failures,
    );
    assert(
      initialState.interactiveObjectRgbPointCount === 209_479,
      `interactiveObjectRgbPointCount=${initialState.interactiveObjectRgbPointCount}`,
      failures,
    );
    assert(initialState.visualCount === 1_512_870, `visualCount=${initialState.visualCount}`, failures);
    assert(initialState.colliderFaces === 1_265_671, `static colliderFaces=${initialState.colliderFaces}`, failures);
    assert(initialState.interactiveObjectReadyCount === 5, "not all visual objects are ready", failures);
    assert(initialState.objectColliderCount === 4, `objectColliderCount=${initialState.objectColliderCount}`, failures);
    assert(collisionWorld.objectColliderReadyCount === 4, "not all object colliders are ready", failures);
    assert(collisionWorld.degradedCount === 0, `degraded colliders=${collisionWorld.degradedCount}`, failures);
    assert(collisionWorld.errors.length === 0, `collider load errors=${JSON.stringify(collisionWorld.errors)}`, failures);
    const pillowCollisionState = collisionWorld.objects.find((item) => item.id === pillowId);
    assert(pillowCollisionState?.ready === false, "pillow incorrectly reports collider ready", failures);
    assert(pillowCollisionState?.mode === "none", `pillow collision mode=${pillowCollisionState?.mode}`, failures);
    assert(pillowCollisionState?.faces === 0, `pillow collision faces=${pillowCollisionState?.faces}`, failures);
    assert(pillowCollisionState?.bounds == null, "pillow unexpectedly has collision bounds", failures);

    await page.evaluate(() => window.__setObjectColliderDebugVisible(false));
    const pillowLocation = await page.evaluate(() => window.__askScene("枕头在哪里？"));
    await page.waitForTimeout(1_200);
    const pillowLocationScreenshot = path.join(screenshotDir, "pillow-location-query.png");
    await page.screenshot({ path: pillowLocationScreenshot, fullPage: true });
    const pillowAppearance = await page.evaluate(() => window.__askScene("枕头长什么样子？"));
    await page.waitForTimeout(1_200);
    const pillowAppearanceScreenshot = path.join(screenshotDir, "pillow-appearance-query.png");
    await page.screenshot({ path: pillowAppearanceScreenshot, fullPage: true });
    const pillowFocus = await page.evaluate(() => window.__visualPhysicsDemoState);
    assert(pillowLocation.status === "resolved", `pillow location status=${pillowLocation.status}`, failures);
    assert(pillowLocation.entityId === "sam3_pillow_01", `pillow location entity=${pillowLocation.entityId}`, failures);
    assert(pillowLocation.focusEntityId === "sam3_pillow_01", "pillow location did not request focus", failures);
    assert(pillowLocation.answer.includes("床上"), `pillow location answer=${pillowLocation.answer}`, failures);
    assert(pillowAppearance.status === "resolved", `pillow appearance status=${pillowAppearance.status}`, failures);
    assert(pillowAppearance.entityId === "sam3_pillow_01", `pillow appearance entity=${pillowAppearance.entityId}`, failures);
    assert(pillowAppearance.focusEntityId === "sam3_pillow_01", "pillow appearance did not request focus", failures);
    assert(
      pillowAppearance.answer.includes("三只相接")
        && pillowAppearance.answer.includes("白色/灰白色")
        && !pillowAppearance.answer.includes("浅绿色"),
      `pillow appearance answer=${pillowAppearance.answer}`,
      failures,
    );
    assert(
      pillowFocus.sceneQaResolvedObjectId === "sam3_pillow_01",
      `resolved scene entity=${pillowFocus.sceneQaResolvedObjectId}`,
      failures,
    );
    await page.evaluate(() => window.__setObjectColliderDebugVisible(true));

    const sceneCognition = {
      pillow: {
        location: pillowLocation,
        appearance: pillowAppearance,
        selectedEntityId: pillowFocus.sceneQaResolvedObjectId,
        screenshots: {
          location: pillowLocationScreenshot,
          appearance: pillowAppearanceScreenshot,
        },
      },
    };

    const objects = [];
    for (const objectId of collidableObjectIds) {
      const inspection = await page.evaluate((id) => window.__inspectInteractiveObject(id), objectId);
      const centerError = vectorDistance(inspection.collisionBounds.center, inspection.proxyWorldBounds.center);
      const sizeError = relativeVectorError(inspection.collisionBounds.size, inspection.proxyWorldBounds.size);
      const rayHits = await page.evaluate(({ id, bounds }) => {
        const axes = [
          { axis: "x", sign: 1 }, { axis: "x", sign: -1 },
          { axis: "y", sign: 1 }, { axis: "y", sign: -1 },
          { axis: "z", sign: 1 }, { axis: "z", sign: -1 },
        ];
        return axes.map(({ axis, sign }) => {
          const origin = { ...bounds.center };
          origin[axis] += sign * (bounds.size[axis] * 0.5 + 0.35);
          const direction = { x: 0, y: 0, z: 0 };
          direction[axis] = -sign;
          const hit = window.__probeCollisionRay(
            [origin.x, origin.y, origin.z],
            [direction.x, direction.y, direction.z],
            bounds.size[axis] + 1,
          );
          return { axis, sign, hit, matched: hit?.objectId === id };
        });
      }, { id: objectId, bounds: inspection.collisionBounds });
      const rayHit = rayHits.find((entry) => entry.matched) || null;
      assert(inspection.collisionMode === "glb", `${objectId}: collisionMode=${inspection.collisionMode}`, failures);
      assert(inspection.collisionFaces > 0, `${objectId}: zero collision faces`, failures);
      assert(centerError <= 0.015, `${objectId}: collider/proxy center error=${centerError}`, failures);
      assert(sizeError <= 0.015, `${objectId}: collider/proxy size error=${sizeError}`, failures);
      assert(Boolean(rayHit), `${objectId}: no direct collision ray matched`, failures);
      objects.push({
        id: objectId,
        collisionMode: inspection.collisionMode,
        collisionFaces: inspection.collisionFaces,
        collisionBounds: inspection.collisionBounds,
        gaussianBounds: inspection.splatWorldBounds,
        proxyBounds: inspection.proxyWorldBounds,
        centerError: rounded(centerError, 6),
        maximumRelativeSizeError: rounded(sizeError, 6),
        directRayHit: rayHit,
      });
    }

    const pillowInspection = await page.evaluate((id) => window.__inspectInteractiveObject(id), pillowId);
    const pillowCenterError = vectorDistance(
      pillowInspection.splatWorldBounds.center,
      pillowInspection.proxyWorldBounds.center,
    );
    const pillowSizeError = relativeVectorError(
      pillowInspection.splatWorldBounds.size,
      pillowInspection.proxyWorldBounds.size,
    );
    const pillowDefinition = objectDefinitions.find((item) => item.id === pillowId);
    const pillowGroupCenter = {
      x: pillowInspection.groupMatrixWorld[12],
      y: pillowInspection.groupMatrixWorld[13],
      z: pillowInspection.groupMatrixWorld[14],
    };
    const pillowPivotError = vectorDistance(pillowGroupCenter, pillowInspection.proxyWorldBounds.center);
    const pillowAcceptedBoxSizeError = relativeVectorError(
      pillowInspection.proxyWorldBounds.size,
      {
        x: pillowDefinition.bbox.extent[0],
        y: pillowDefinition.bbox.extent[1],
        z: pillowDefinition.bbox.extent[2],
      },
    );
    assert(pillowInspection.visualReady === true, "pillow RGB point visual is not ready", failures);
    assert(pillowInspection.visualKind === "rgb-points", `pillow visualKind=${pillowInspection.visualKind}`, failures);
    assert(pillowInspection.colliderReady === false, "pillow incorrectly reports colliderReady", failures);
    assert(pillowInspection.collisionMode === "none", `pillow collisionMode=${pillowInspection.collisionMode}`, failures);
    assert(pillowInspection.collisionFaces === 0, `pillow collisionFaces=${pillowInspection.collisionFaces}`, failures);
    assert(pillowInspection.collisionBounds == null, "pillow unexpectedly exposes collisionBounds", failures);
    assert(pillowPivotError <= 0.00001, `pillow accepted pivot error=${pillowPivotError}`, failures);
    assert(
      pillowAcceptedBoxSizeError <= 0.00001,
      `pillow accepted bbox size error=${pillowAcceptedBoxSizeError}`,
      failures,
    );
    assert(
      pillowCarve.anchorSha256 === pillowDefinition.visual.sha256
        && pillowCarve.anchorPointCount === pillowDefinition.visual.vertexCount,
      "pillow component does not load the accepted point-cloud hash/count",
      failures,
    );
    assert(
      pillowCarve.anchorInsideAcceptedAabbRatio >= 0.97,
      `pillow accepted-AABB point coverage=${pillowCarve.anchorInsideAcceptedAabbRatio}`,
      failures,
    );
    objects.push({
      id: pillowId,
      visualKind: pillowInspection.visualKind,
      visualReady: pillowInspection.visualReady,
      colliderReady: pillowInspection.colliderReady,
      collisionMode: pillowInspection.collisionMode,
      collisionFaces: pillowInspection.collisionFaces,
      collisionBounds: pillowInspection.collisionBounds,
      splatWorldBounds: pillowInspection.splatWorldBounds,
      proxyWorldBounds: pillowInspection.proxyWorldBounds,
      acceptedPivotError: rounded(pillowPivotError, 9),
      acceptedBoundingBoxSizeError: rounded(pillowAcceptedBoxSizeError, 9),
      rawPointBoundsVsAcceptedCenterError: rounded(pillowCenterError, 6),
      rawPointBoundsVsAcceptedMaximumRelativeSizeDifference: rounded(pillowSizeError, 6),
      acceptedAabbPointCoverage: pillowCarve.anchorInsideAcceptedAabbRatio,
      rawPointTailCaveat: "The accepted AABB is robust; 2.76% of accepted raw points form tails outside it and remain visual-only.",
      acceptedPointCount: 209_479,
    });

    const overviewScreenshot = path.join(screenshotDir, "overview-collider-overlay.png");
    await page.evaluate(() => window.__setCameraPreset("reference"));
    await page.waitForTimeout(900);
    await page.screenshot({ path: overviewScreenshot, fullPage: true });
    const objectScreenshots = [];
    for (const objectId of objectIds) {
      await page.evaluate((id) => window.__setObjectColliderDebugVisible(true, id), objectId);
      await page.evaluate((id) => window.__focusSceneEntity(id), objectId);
      await page.waitForTimeout(1_200);
      const screenshotPath = path.join(screenshotDir, `${objectId}-context-overlay.png`);
      await page.screenshot({ path: screenshotPath, fullPage: true });
      objectScreenshots.push({ objectId, context: screenshotPath, visualOnly: null, solo: null });
    }
    await page.locator("#toggleStaticVisual").click();
    await page.locator("#toggleObjectSolo").click();
    for (const screenshot of objectScreenshots) {
      await page.evaluate((id) => window.__focusSceneEntity(id), screenshot.objectId);
      await page.waitForTimeout(700);
      await page.evaluate(() => window.__setObjectColliderDebugVisible(false));
      screenshot.visualOnly = path.join(screenshotDir, `${screenshot.objectId}-visual-only.png`);
      await page.screenshot({ path: screenshot.visualOnly, fullPage: true });
      await page.evaluate((id) => window.__setObjectColliderDebugVisible(true, id), screenshot.objectId);
      screenshot.solo = path.join(screenshotDir, `${screenshot.objectId}-solo-overlay.png`);
      await page.screenshot({ path: screenshot.solo, fullPage: true });
    }
    await page.locator("#toggleObjectSolo").click();
    await page.locator("#toggleStaticVisual").click();
    await page.evaluate(() => window.__setObjectColliderDebugVisible(true));

    const spinId = "sam3_plant_01";
    await page.evaluate((id) => window.__focusSceneEntity(id), spinId);
    await page.waitForTimeout(1_000);
    const spinPoint = await page.evaluate((id) => window.__getInteractiveObjectScreenPoint(id), spinId);
    const turnsBefore = await page.evaluate(() => window.__visualPhysicsDemoState.interactiveObjectTurns);
    await page.mouse.dblclick(spinPoint.x, spinPoint.y, { delay: 45 });
    await page.waitForFunction((turns) => window.__visualPhysicsDemoState.interactiveObjectTurns > turns, turnsBefore, {
      timeout: 8_000,
    });
    const spinState = await page.evaluate(() => window.__visualPhysicsDemoState);
    assert(spinState.selectedInteractiveObject === spinId, "double-click selected the wrong object", failures);

    const dragId = "sam3_nightstand_01";
    await page.evaluate((id) => window.__focusSceneEntity(id), dragId);
    await page.waitForTimeout(1_000);
    const dragPoint = await page.evaluate((id) => window.__getInteractiveObjectScreenPoint(id), dragId);
    const dragBefore = await page.evaluate((id) => window.__inspectInteractiveObject(id).groupMatrixWorld, dragId);
    await page.mouse.move(dragPoint.x, dragPoint.y);
    await page.mouse.down();
    await page.mouse.move(dragPoint.x + 82, dragPoint.y, { steps: 8 });
    await page.mouse.up();
    const dragAfter = await page.evaluate((id) => ({
      matrix: window.__inspectInteractiveObject(id).groupMatrixWorld,
      activeDrag: window.__visualPhysicsDemoState.activeObjectDrag,
    }), dragId);
    const dragChanged = JSON.stringify(dragBefore) !== JSON.stringify(dragAfter.matrix);
    assert(dragChanged, "pointer drag did not rotate the component", failures);
    assert(dragAfter.activeDrag == null, "pointer drag remained active after mouseup", failures);
    await page.evaluate((id) => window.__setInteractiveObjectYaw(id, 0), dragId);

    await page.evaluate((id) => window.__focusSceneEntity(id), pillowId);
    await page.waitForTimeout(1_000);
    const pillowDragPoint = await page.evaluate(
      (id) => window.__getInteractiveObjectScreenPoint(id),
      pillowId,
    );
    const pillowBeforeDrag = await page.evaluate(
      (id) => window.__inspectInteractiveObject(id),
      pillowId,
    );
    const pillowCollisionTargetsBefore = await page.evaluate(
      () => window.__inspectCollisionWorld().targetCount,
    );
    await page.mouse.move(pillowDragPoint.x, pillowDragPoint.y);
    await page.mouse.down();
    await page.mouse.move(pillowDragPoint.x + 96, pillowDragPoint.y, { steps: 10 });
    await page.mouse.up();
    const pillowAfterDrag = await page.evaluate((id) => ({
      inspection: window.__inspectInteractiveObject(id),
      activeDrag: window.__visualPhysicsDemoState.activeObjectDrag,
      collisionWorld: window.__inspectCollisionWorld(),
    }), pillowId);
    const pillowGroupChanged = JSON.stringify(pillowBeforeDrag.groupMatrixWorld)
      !== JSON.stringify(pillowAfterDrag.inspection.groupMatrixWorld);
    const pillowVisualChanged = JSON.stringify(pillowBeforeDrag.splatMatrixWorld)
      !== JSON.stringify(pillowAfterDrag.inspection.splatMatrixWorld);
    const pillowProxyChanged = JSON.stringify(pillowBeforeDrag.proxyMatrixWorld)
      !== JSON.stringify(pillowAfterDrag.inspection.proxyMatrixWorld);
    assert(pillowGroupChanged, "pillow pointer drag did not rotate the parent group", failures);
    assert(pillowVisualChanged, "pillow RGB point visual did not follow drag", failures);
    assert(pillowProxyChanged, "pillow selection bbox did not follow drag", failures);
    assert(pillowAfterDrag.activeDrag == null, "pillow pointer drag remained active", failures);
    assert(
      pillowAfterDrag.collisionWorld.targetCount === pillowCollisionTargetsBefore,
      "pillow drag changed the collision target count",
      failures,
    );
    assert(
      pillowAfterDrag.collisionWorld.objectColliderReadyCount === 4,
      "pillow drag changed object collider readiness",
      failures,
    );
    const pillowDragScreenshot = path.join(screenshotDir, "sam3_pillow_01-dragged.png");
    await page.screenshot({ path: pillowDragScreenshot, fullPage: true });
    await page.evaluate((id) => window.__setInteractiveObjectYaw(id, 0), pillowId);
    const pillowBeforeSpin = await page.evaluate((id) => ({
      inspection: window.__inspectInteractiveObject(id),
      turns: window.__visualPhysicsDemoState.interactiveObjectTurns,
    }), pillowId);
    const pillowSpinPoint = await page.evaluate(
      (id) => window.__getInteractiveObjectScreenPoint(id),
      pillowId,
    );
    await page.mouse.dblclick(pillowSpinPoint.x, pillowSpinPoint.y, { delay: 45 });
    await page.waitForFunction(
      (turns) => window.__visualPhysicsDemoState.interactiveObjectTurns > turns,
      pillowBeforeSpin.turns,
      { timeout: 8_000 },
    );
    const pillowAfterSpin = await page.evaluate((id) => ({
      inspection: window.__inspectInteractiveObject(id),
      turns: window.__visualPhysicsDemoState.interactiveObjectTurns,
      collisionWorld: window.__inspectCollisionWorld(),
    }), pillowId);
    const pillowSpinReturnedToStart = JSON.stringify(pillowBeforeSpin.inspection.groupMatrixWorld)
      === JSON.stringify(pillowAfterSpin.inspection.groupMatrixWorld);
    assert(pillowSpinReturnedToStart, "pillow 360 spin did not return to its start transform", failures);
    assert(
      pillowAfterSpin.collisionWorld.objectColliderReadyCount === 4
        && pillowAfterSpin.collisionWorld.objects.find((item) => item.id === pillowId)?.mode === "none",
      "pillow spin violated the visual-only collision boundary",
      failures,
    );
    const pillowVisualOnlyInteraction = {
      pointerDrag: {
        objectId: pillowId,
        parentGroupMatrixChanged: pillowGroupChanged,
        pointVisualMatrixChanged: pillowVisualChanged,
        selectionBoundsMatrixChanged: pillowProxyChanged,
        released: pillowAfterDrag.activeDrag == null,
        collisionTargetCountUnchanged: pillowAfterDrag.collisionWorld.targetCount === pillowCollisionTargetsBefore,
        passed: pillowGroupChanged && pillowVisualChanged && pillowProxyChanged
          && pillowAfterDrag.activeDrag == null,
        screenshot: pillowDragScreenshot,
      },
      doubleClick360: {
        objectId: pillowId,
        turnsBefore: pillowBeforeSpin.turns,
        turnsAfter: pillowAfterSpin.turns,
        returnedToStartTransform: pillowSpinReturnedToStart,
        collisionModeAfter: pillowAfterSpin.collisionWorld.objects.find((item) => item.id === pillowId)?.mode,
        passed: pillowAfterSpin.turns > pillowBeforeSpin.turns && pillowSpinReturnedToStart,
      },
    };

    const robotCollisions = [];
    for (const objectId of collidableObjectIds) {
      const prepared = await page.evaluate((id) => window.__prepareRobotCollisionTest(id), objectId);
      let step = null;
      if (prepared.prepared) {
        for (let attempt = 0; attempt < 3; attempt += 1) {
          step = await page.evaluate((id) => window.__stepRobotTowardObject(id, 0.65), objectId);
          if (step.blocked && step.lastCollisionObjectId === objectId) break;
        }
      }
      const passed = Boolean(prepared.prepared && step?.blocked && step?.lastCollisionObjectId === objectId);
      assert(passed, `${objectId}: robot collision did not block on the intended object`, failures);
      robotCollisions.push({
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
      });
    }

    const fpsSamples = [];
    for (let index = 0; index < 5; index += 1) {
      await page.waitForTimeout(1_050);
      fpsSamples.push(await page.evaluate(() => window.__visualPhysicsDemoState.fps));
    }
    const averageFps = fpsSamples.reduce((sum, value) => sum + value, 0) / fpsSamples.length;
    const memory = await page.evaluate(() => {
      const info = performance.memory;
      return info ? {
        usedJSHeapSize: info.usedJSHeapSize,
        totalJSHeapSize: info.totalJSHeapSize,
        jsHeapSizeLimit: info.jsHeapSizeLimit,
      } : null;
    });
    assert(averageFps >= 18, `average FPS ${averageFps.toFixed(1)} is below 18`, failures);

    await context.close();
    const mobileContext = await browser.newContext({
      viewport: { width: 390, height: 844 },
      deviceScaleFactor: 1,
      isMobile: true,
      hasTouch: true,
    });
    const mobilePage = await mobileContext.newPage();
    mobilePage.on("console", (message) => {
      if (["error", "warning"].includes(message.type())) {
        consoleMessages.push({ type: message.type(), text: message.text(), viewport: "mobile" });
      }
    });
    mobilePage.on("pageerror", (error) => pageErrors.push(`mobile: ${error.message}`));
    mobilePage.on("requestfailed", (request) => {
      requestFailures.push({
        url: request.url(),
        error: request.failure()?.errorText || "unknown",
        viewport: "mobile",
      });
    });
    await mobilePage.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
    await mobilePage.waitForFunction(() => {
      const state = window.__visualPhysicsDemoState;
      return state?.visualReady
        && state?.colliderReady
        && state?.sceneQaReady
        && state?.interactiveObjectReadyCount === 5
        && state?.objectColliderReadyCount === 4;
    }, null, { timeout: 300_000 });
    const mobileAppearance = await mobilePage.evaluate(() => window.__askScene("枕头长什么样子？"));
    await mobilePage.waitForTimeout(1_200);
    const mobileScreenshot = path.join(screenshotDir, "mobile-390x844-pillow-appearance.png");
    await mobilePage.screenshot({ path: mobileScreenshot, fullPage: true });
    const mobileLayout = await mobilePage.evaluate(() => {
      const rect = (element) => {
        const value = element.getBoundingClientRect();
        return {
          left: value.left,
          top: value.top,
          right: value.right,
          bottom: value.bottom,
          width: value.width,
          height: value.height,
        };
      };
      const canvas = document.querySelector("#sceneCanvas");
      const hud = document.querySelector(".hud");
      const answer = document.querySelector("#sceneQaAnswer");
      const queryRow = document.querySelector(".scene-qa-row");
      const metrics = document.querySelector(".metrics");
      const statusCard = document.querySelector(".status-card");
      const controls = document.querySelector(".controls");
      const canvasRect = rect(canvas);
      const hudRect = rect(hud);
      const answerRect = rect(answer);
      const queryRowRect = rect(queryRow);
      const metricsRect = rect(metrics);
      const statusCardRect = rect(statusCard);
      const controlsRect = rect(controls);
      return {
        viewport: [window.innerWidth, window.innerHeight],
        canvas: canvasRect,
        canvasBackingStore: [canvas.width, canvas.height],
        hud: {
          ...hudRect,
          rightInset: window.innerWidth - hudRect.right,
          scrollWidth: hud.scrollWidth,
          clientWidth: hud.clientWidth,
          scrollHeight: hud.scrollHeight,
          clientHeight: hud.clientHeight,
        },
        body: {
          scrollWidth: document.body.scrollWidth,
          clientWidth: document.body.clientWidth,
          htmlScrollWidth: document.documentElement.scrollWidth,
          htmlClientWidth: document.documentElement.clientWidth,
        },
        answer: {
          ...answerRect,
          text: answer.textContent,
          scrollWidth: answer.scrollWidth,
          clientWidth: answer.clientWidth,
          scrollHeight: answer.scrollHeight,
          clientHeight: answer.clientHeight,
        },
        queryRow: queryRowRect,
        metrics: metricsRect,
        statusCard: statusCardRect,
        controls: controlsRect,
        nonOverlap: {
          queryRowBeforeAnswer: queryRowRect.bottom <= answerRect.top + 0.5,
          answerBeforeMetrics: answerRect.bottom <= metricsRect.top + 0.5,
          statusBeforeControls: statusCardRect.bottom <= controlsRect.top + 0.5,
        },
      };
    });
    const responsiveChecks = {
      appearanceResolved: mobileAppearance.status === "resolved",
      detailedAnswer: mobileAppearance.answer.includes("三只相接")
        && mobileAppearance.answer.includes("白色/灰白色")
        && !mobileAppearance.answer.includes("浅绿色"),
      exactViewport: mobileLayout.viewport[0] === 390 && mobileLayout.viewport[1] === 844,
      exactCanvasCssSize: Math.abs(mobileLayout.canvas.width - 390) <= 0.5
        && Math.abs(mobileLayout.canvas.height - 844) <= 0.5,
      exactHudInsetsAndWidth: Math.abs(mobileLayout.hud.left - 12) <= 0.5
        && Math.abs(mobileLayout.hud.rightInset - 12) <= 0.5
        && Math.abs(mobileLayout.hud.width - 366) <= 0.5,
      noHorizontalOverflow: mobileLayout.body.scrollWidth <= 390
        && mobileLayout.body.htmlScrollWidth <= 390
        && mobileLayout.hud.scrollWidth <= mobileLayout.hud.clientWidth,
      answerHasNoOwnOverflow: mobileLayout.answer.scrollWidth <= mobileLayout.answer.clientWidth
        && mobileLayout.answer.scrollHeight <= mobileLayout.answer.clientHeight,
      controlsDoNotOverlap: Object.values(mobileLayout.nonOverlap).every(Boolean),
    };
    assert(responsiveChecks.appearanceResolved, `mobile pillow status=${mobileAppearance.status}`, failures);
    assert(
      responsiveChecks.detailedAnswer,
      `mobile pillow answer=${mobileAppearance.answer}`,
      failures,
    );
    assert(
      responsiveChecks.exactViewport,
      `mobile viewport=${mobileLayout.viewport.join("x")}`,
      failures,
    );
    assert(
      responsiveChecks.exactCanvasCssSize,
      `mobile canvas=${mobileLayout.canvas.width}x${mobileLayout.canvas.height}`,
      failures,
    );
    assert(
      responsiveChecks.exactHudInsetsAndWidth,
      `mobile HUD left/right/width=${mobileLayout.hud.left}/${mobileLayout.hud.rightInset}/${mobileLayout.hud.width}`,
      failures,
    );
    assert(
      responsiveChecks.noHorizontalOverflow,
      `mobile horizontal overflow body/html/hud=${mobileLayout.body.scrollWidth}/${mobileLayout.body.htmlScrollWidth}/${mobileLayout.hud.scrollWidth}`,
      failures,
    );
    assert(
      responsiveChecks.answerHasNoOwnOverflow,
      `mobile answer overflow=${mobileLayout.answer.scrollWidth}/${mobileLayout.answer.clientWidth} ${mobileLayout.answer.scrollHeight}/${mobileLayout.answer.clientHeight}`,
      failures,
    );
    assert(
      responsiveChecks.controlsDoNotOverlap,
      `mobile answer/control overlap=${JSON.stringify(mobileLayout.nonOverlap)}`,
      failures,
    );
    const responsive = {
      status: Object.values(responsiveChecks).every(Boolean) ? "passed" : "failed",
      checks: responsiveChecks,
      freshPageLoad: true,
      viewport: [390, 844],
      appearance: mobileAppearance,
      layout: mobileLayout,
      screenshot: mobileScreenshot,
    };
    await mobileContext.close();

    assert(pageErrors.length === 0, `page errors=${pageErrors.join(" | ")}`, failures);
    assert(requestFailures.length === 0, `request failures=${JSON.stringify(requestFailures)}`, failures);
    const consoleErrors = consoleMessages.filter((message) => message.type === "error");
    assert(consoleErrors.length === 0, `console errors=${JSON.stringify(consoleErrors)}`, failures);

    report = {
      schemaVersion: 1,
      status: failures.length ? "failed" : "candidate",
      automatedGate: failures.length ? "failed" : "passed",
      placementReview: { status: "pending", reviewer: null, reviewedAt: null },
      startedAt,
      finishedAt: new Date().toISOString(),
      durationSeconds: rounded((Date.now() - startedMs) / 1000, 2),
      url,
      browser: { engine: "chromium", channel: "chrome", headless: true, viewport: [1440, 900] },
      manifestPath,
      manifestSha256: sha256File(manifestPath),
      counts: {
        staticGaussians: initialState.staticVisualCount,
        objectGaussians: initialState.interactiveObjectGaussianCount,
        objectRgbPoints: initialState.interactiveObjectRgbPointCount,
        totalVisualPrimitives: initialState.visualCount,
        visualObjectsReady: initialState.interactiveObjectReadyCount,
        visualObjectsExpected: initialState.interactiveObjectCount,
        objectCollidersReady: initialState.objectColliderReadyCount,
        objectCollidersExpected: initialState.objectColliderCount,
        staticColliderFaces: initialState.colliderFaces,
        objectColliderFaces: collisionWorld.objectColliderFaces,
        collisionTargets: collisionWorld.targetCount,
      },
      collisionWorld,
      staticVisualCarve: pillowCarve,
      sceneCognition,
      objects,
      interactions: {
        doubleClick360: { objectId: spinId, turnsBefore, turnsAfter: spinState.interactiveObjectTurns, passed: true },
        pointerDrag: { objectId: dragId, matrixChanged: dragChanged, released: dragAfter.activeDrag == null },
        pillowVisualOnly: pillowVisualOnlyInteraction,
        robotCollisions,
      },
      performance: {
        fpsSamples,
        averageFps: rounded(averageFps, 2),
        minimumFps: Math.min(...fpsSamples),
        memory,
      },
      responsive,
      console: { messages: consoleMessages, pageErrors, requestFailures },
      screenshots: {
        overview: overviewScreenshot,
        sceneCognition: sceneCognition.pillow.screenshots,
        pillowDrag: pillowDragScreenshot,
        mobile: mobileScreenshot,
        objects: objectScreenshots,
      },
      failures,
    };
  } catch (error) {
    report = {
      schemaVersion: 1,
      status: "failed",
      automatedGate: "failed",
      placementReview: { status: "pending", reviewer: null, reviewedAt: null },
      startedAt,
      finishedAt: new Date().toISOString(),
      durationSeconds: rounded((Date.now() - startedMs) / 1000, 2),
      url,
      manifestPath,
      manifestSha256: sha256File(manifestPath),
      console: { messages: consoleMessages, pageErrors, requestFailures },
      failures: [...failures, error?.stack || String(error)],
    };
  } finally {
    await browser.close();
  }

  report = portableReport(report, worldDir);
  writeJson(browserReportPath, report);
  writeJson(exampleBrowserReport, report);
  console.log(JSON.stringify(report, null, 2));
  if (report.automatedGate !== "passed") process.exitCode = 1;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const worldDir = path.resolve(options["world-dir"] || DEFAULT_WORLD_DIR);
  const browserReportPath = path.resolve(options.report || path.join(worldDir, "qa/browser-qa.json"));
  const exampleBrowserReport = path.resolve(
    options["example-report"] || path.join(DEFAULT_EXAMPLE_DIR, "browser-qa.json"),
  );
  if (options["finalize-placement"] === "passed") {
    promotePlacementGate({ worldDir, browserReportPath, exampleBrowserReport });
    return;
  }
  if (options["finalize-placement"]) {
    throw new Error("--finalize-placement only accepts passed");
  }
  const screenshotDir = path.resolve(options["screenshots-dir"] || path.join(worldDir, "qa/screenshots"));
  const url = options.url
    || "http://127.0.0.1:4175/?manifest=/worlds/bedroom4/manifest.json&allowCandidateColliders=1&collisionDebug=on";
  await runBrowserQa({ url, worldDir, browserReportPath, exampleBrowserReport, screenshotDir });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
