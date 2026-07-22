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
const defaultManifest = path.join(publicRoot, "worlds/bedroom4/manifest.lamp-candidate.json");
const defaultArtifactRoot = path.join(
  repoRoot,
  "examples/bedroom4/completion/lamps_trellis_seed43/browser_qa",
);
const defaultReport = path.join(defaultArtifactRoot, "browser_qa_report.json");
const defaultUrl = "http://127.0.0.1:4185/";

const UNIFIED_IDS = Object.freeze([
  "sam3_pillow_front",
  "sam3_pillow_left",
  "sam3_pillow_right",
  "sam3_bed_01",
  "sam3_lamp_01",
  "sam3_lamp_02",
]);
const PILLOW_IDS = Object.freeze(UNIFIED_IDS.slice(0, 3));
const LAMP_IDS = Object.freeze(UNIFIED_IDS.slice(4));

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) throw new Error(`Unknown positional argument: ${arg}`);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Missing value for ${arg}`);
    result[arg.slice(2)] = value;
    index += 1;
  }
  return result;
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

function sha256File(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function fileDescriptor(filePath) {
  const stats = fs.statSync(filePath);
  return { path: filePath, size: stats.size, sha256: sha256File(filePath) };
}

function manifestUrlForPath(manifestPath) {
  const relative = path.relative(publicRoot, manifestPath);
  if (!relative.startsWith("..") && !path.isAbsolute(relative)) {
    return `./${relative.split(path.sep).join("/")}`;
  }
  return `/@fs/${manifestPath}`;
}

function runtimeUrl(baseUrl, manifestPath, runtimeManifestUrl = null) {
  const url = new URL(baseUrl);
  const manifestUrl = runtimeManifestUrl || manifestUrlForPath(manifestPath);
  const resolvedManifest = new URL(manifestUrl, url);
  if (resolvedManifest.origin !== url.origin) {
    throw new Error(`Runtime manifest must be same-origin with the page: ${manifestUrl}`);
  }
  url.searchParams.set("manifest", manifestUrl);
  return url.href;
}

function matrixDelta(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return null;
  return Math.max(...left.map((value, index) => Math.abs(Number(value) - Number(right[index]))));
}

function record(condition, message, failures) {
  if (!condition) failures.push(message);
  return Boolean(condition);
}

function listenForDiagnostics(page) {
  const diagnostics = { pageErrors: [], consoleErrors: [], warnings: [], requestFailures: [] };
  const successfulResponseUrls = new Set();
  Object.defineProperty(diagnostics, "successfulResponseUrls", {
    value: successfulResponseUrls,
    enumerable: false,
  });
  page.on("pageerror", (error) => diagnostics.pageErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") diagnostics.consoleErrors.push(message.text());
    if (message.type() === "warning") diagnostics.warnings.push(message.text());
  });
  page.on("response", (response) => {
    if (response.ok()) successfulResponseUrls.add(response.url());
  });
  page.on("requestfailed", (request) => diagnostics.requestFailures.push({
    url: request.url(),
    error: request.failure()?.errorText || "unknown",
  }));
  return diagnostics;
}

function finalizeDiagnostics(diagnostics) {
  const recoveredRequestFailures = diagnostics.requestFailures.filter(
    (failure) => diagnostics.successfulResponseUrls.has(failure.url),
  );
  const requestFailures = diagnostics.requestFailures.filter(
    (failure) => !diagnostics.successfulResponseUrls.has(failure.url),
  );
  const networkConsoleErrors = diagnostics.consoleErrors.filter(
    (message) => /^Failed to load resource: net::ERR_/u.test(message),
  );
  const otherConsoleErrors = diagnostics.consoleErrors.filter(
    (message) => !/^Failed to load resource: net::ERR_/u.test(message),
  );
  const allNetworkFailuresRecovered = requestFailures.length === 0
    && recoveredRequestFailures.length > 0;
  return {
    pageErrors: diagnostics.pageErrors,
    consoleErrors: allNetworkFailuresRecovered
      ? otherConsoleErrors
      : [...otherConsoleErrors, ...networkConsoleErrors],
    warnings: diagnostics.warnings,
    requestFailures,
    recoveredConsoleErrors: allNetworkFailuresRecovered ? networkConsoleErrors : [],
    recoveredRequestFailures,
  };
}

async function waitForReady(page, contract) {
  await page.waitForFunction((expected) => {
    const state = window.__visualPhysicsDemoState;
    return Boolean(state?.error) || Boolean(
      state?.visualReady
      && state?.colliderReady
      && state?.sceneQaReady
      && state?.interactiveObjectReadyCount === expected.objectCount
      && state?.objectColliderReadyCount === expected.objectColliderCount,
    );
  }, contract, { timeout: 300_000 });
  const state = await page.evaluate(() => window.__visualPhysicsDemoState);
  if (state.error) throw new Error(`Runtime load failed: ${state.error}`);
}

async function inspectCanvasPixels(page) {
  return page.locator("#sceneCanvas").evaluate(async (canvas) => {
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const sample = document.createElement("canvas");
    sample.width = 96;
    sample.height = 96;
    const context = sample.getContext("2d", { willReadFrequently: true });
    context.drawImage(canvas, 0, 0, sample.width, sample.height);
    const data = context.getImageData(0, 0, sample.width, sample.height).data;
    const colors = new Set();
    let nonDarkPixels = 0;
    let minimumLuma = 255;
    let maximumLuma = 0;
    for (let index = 0; index < data.length; index += 4) {
      const luma = 0.2126 * data[index] + 0.7152 * data[index + 1] + 0.0722 * data[index + 2];
      if (luma > 8) nonDarkPixels += 1;
      minimumLuma = Math.min(minimumLuma, luma);
      maximumLuma = Math.max(maximumLuma, luma);
      colors.add(`${data[index] >> 4}:${data[index + 1] >> 4}:${data[index + 2] >> 4}`);
    }
    return {
      sampleSize: [sample.width, sample.height],
      backingStore: [canvas.width, canvas.height],
      nonDarkPixels,
      quantizedColorCount: colors.size,
      lumaRange: Number((maximumLuma - minimumLuma).toFixed(3)),
      nonblank: nonDarkPixels >= 64 && colors.size >= 8 && maximumLuma - minimumLuma >= 8,
    };
  });
}

async function inspectLayout(page) {
  return page.evaluate(() => ({
    viewport: [window.innerWidth, window.innerHeight],
    bodyScrollWidth: document.body.scrollWidth,
    documentScrollWidth: document.documentElement.scrollWidth,
    noHorizontalOverflow: document.body.scrollWidth <= window.innerWidth
      && document.documentElement.scrollWidth <= window.innerWidth,
    qaVisible: Boolean(document.querySelector(".scene-qa-row")?.getBoundingClientRect().height),
    canvasVisible: Boolean(document.querySelector("#sceneCanvas")?.getBoundingClientRect().height),
  }));
}

async function screenshot(page, outputDir, name) {
  fs.mkdirSync(outputDir, { recursive: true });
  const outputPath = path.join(outputDir, `${name}.png`);
  await page.screenshot({ path: outputPath, fullPage: false });
  return fileDescriptor(outputPath);
}

async function findObjectHit(page, objectId) {
  return page.evaluate((id) => {
    const seed = window.__getInteractiveObjectScreenPoint(id);
    const canvas = document.querySelector("#sceneCanvas");
    const bounds = canvas?.getBoundingClientRect();
    if (!seed || !bounds) return { seed, point: null, hit: null };
    const inspect = (x, y) => {
      if (x < bounds.left || x > bounds.right || y < bounds.top || y > bounds.bottom) return null;
      if (document.elementFromPoint(x, y) !== canvas) return null;
      const hit = window.__inspectInteractiveObjectHitAt(x, y);
      return hit?.objectId === id ? { point: { x, y }, hit } : null;
    };
    for (let radius = 0; radius <= 220; radius += 10) {
      const count = radius === 0 ? 1 : Math.max(12, Math.ceil(2 * Math.PI * radius / 10));
      for (let index = 0; index < count; index += 1) {
        const angle = count === 1 ? 0 : 2 * Math.PI * index / count;
        const found = inspect(seed.x + Math.cos(angle) * radius, seed.y + Math.sin(angle) * radius);
        if (found) return { seed, ...found };
      }
    }
    return { seed, point: null, hit: null };
  }, objectId);
}

async function askAndInspect(page, question, expectedId) {
  const answer = await page.evaluate((value) => window.__askScene(value), question);
  await page.waitForTimeout(700);
  const snapshot = await page.evaluate((id) => ({
    state: window.__visualPhysicsDemoState,
    inspection: window.__inspectInteractiveObject(id),
    focusVisibility: window.__inspectInteractiveObjectFocusVisibility(id),
  }), expectedId);
  return { question, expectedId, answer, snapshot };
}

async function testPointerDrag(page, objectId) {
  await page.evaluate((id) => window.__focusSceneEntity(id), objectId);
  await page.waitForTimeout(700);
  const target = await findObjectHit(page, objectId);
  const before = await page.evaluate((id) => window.__inspectInteractiveObject(id), objectId);
  if (!target.point) return { target, before, after: null, matrixDelta: null };
  await page.mouse.move(target.point.x, target.point.y);
  await page.mouse.down();
  await page.mouse.move(target.point.x + 84, target.point.y, { steps: 12 });
  await page.mouse.up();
  const after = await page.evaluate((id) => window.__inspectInteractiveObject(id), objectId);
  return {
    target,
    before,
    after,
    matrixDelta: matrixDelta(before.groupMatrixWorld, after.groupMatrixWorld),
    visualCollisionMatrixDelta: matrixDelta(after.splatMatrixWorld, after.collisionMatrixWorld),
  };
}

async function testSpin(page, objectId) {
  await page.evaluate((id) => window.__setInteractiveObjectYaw(id, 0), objectId);
  const before = await page.evaluate((id) => window.__inspectInteractiveObject(id), objectId);
  const start = await page.evaluate((id) => window.__spinInteractiveObject(id), objectId);
  await page.waitForTimeout(350);
  const during = await page.evaluate((id) => window.__inspectInteractiveObject(id), objectId);
  const completed = await page.waitForFunction(
    ({ id, turns }) => window.__inspectInteractiveObject(id)?.turns > turns,
    { id: objectId, turns: before.turns },
    { timeout: 30_000 },
  ).then(() => true).catch(() => false);
  const after = await page.evaluate((id) => window.__inspectInteractiveObject(id), objectId);
  return {
    before,
    start,
    completed,
    during,
    after,
    duringMatrixDelta: matrixDelta(before.groupMatrixWorld, during.groupMatrixWorld),
    returnMatrixDelta: matrixDelta(before.groupMatrixWorld, after.groupMatrixWorld),
    visualCollisionMatrixDelta: matrixDelta(after.splatMatrixWorld, after.collisionMatrixWorld),
  };
}

function expectedContract(manifest) {
  const objectColliderFaces = manifest.interactiveObjects.reduce(
    (sum, object) => sum + Number(object.collision?.asset?.faces || 0),
    0,
  );
  return {
    objectCount: manifest.interactiveObjects.length,
    objectColliderCount: manifest.interactiveObjects.length,
    objectColliderFaces,
    staticVisualCount: manifest.assets.visual.vertexCount,
    staticColliderFaces: manifest.assets.colliderStaticCarved.faceCount,
  };
}

async function runDesktop(browser, url, outputDir, manifest, contract) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const diagnostics = listenForDiagnostics(page);
  const failures = [];
  console.log(JSON.stringify({ stage: "desktop_load", url }));
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
  await waitForReady(page, contract);
  await page.waitForTimeout(1_000);

  const initial = await page.evaluate((ids) => ({
    state: window.__visualPhysicsDemoState,
    collisionWorld: window.__inspectCollisionWorld(),
    interpenetrations: window.__inspectSceneInterpenetrations(),
    inspections: Object.fromEntries(ids.map((id) => [id, window.__inspectInteractiveObject(id)])),
    genericLampResolution: window.__resolveSceneEntity("台灯在哪里？"),
  }), UNIFIED_IDS);
  const canvas = await inspectCanvasPixels(page);
  const layout = await inspectLayout(page);
  const screenshots = { overview: await screenshot(page, outputDir, "desktop_overview") };

  record(initial.state.visualReady === true, "desktop: static 3DGS visual is not ready", failures);
  record(initial.state.colliderReady === true, "desktop: static collider is not ready", failures);
  record(initial.state.sceneQaReady === true, "desktop: Scene QA is not ready", failures);
  record(initial.state.staticVisualCount === contract.staticVisualCount,
    `desktop: static visual count=${initial.state.staticVisualCount}`, failures);
  record(initial.state.colliderFaces === contract.staticColliderFaces,
    `desktop: collider faces=${initial.state.colliderFaces}`, failures);
  record(initial.state.interactiveObjectReadyCount === contract.objectCount,
    `desktop: visual object count=${initial.state.interactiveObjectReadyCount}`, failures);
  record(initial.state.objectColliderReadyCount === contract.objectColliderCount,
    `desktop: collider object count=${initial.state.objectColliderReadyCount}`, failures);
  record(initial.collisionWorld.objectColliderFaces === contract.objectColliderFaces,
    `desktop: object collider faces=${initial.collisionWorld.objectColliderFaces}`, failures);
  record(initial.collisionWorld.degradedCount === 0,
    `desktop: degraded colliders=${initial.collisionWorld.degradedCount}`, failures);
  record(initial.collisionWorld.errors.length === 0,
    `desktop: collider errors=${JSON.stringify(initial.collisionWorld.errors)}`, failures);
  record(canvas.nonblank === true, `desktop: blank canvas=${JSON.stringify(canvas)}`, failures);
  record(layout.noHorizontalOverflow && layout.qaVisible && layout.canvasVisible,
    `desktop: invalid layout=${JSON.stringify(layout)}`, failures);
  record(initial.genericLampResolution?.status === "ambiguous"
    && initial.genericLampResolution?.candidates?.length === 2,
  `desktop: generic lamp query is not ambiguous=${JSON.stringify(initial.genericLampResolution)}`, failures);

  for (const objectId of UNIFIED_IDS) {
    const inspection = initial.inspections[objectId];
    const definition = manifest.interactiveObjects.find((object) => object.id === objectId);
    record(inspection?.visualReady === true, `${objectId}: visual is not ready`, failures);
    record(inspection?.colliderReady === true, `${objectId}: collider is not ready`, failures);
    record(inspection?.visualKind === "mesh", `${objectId}: visualKind=${inspection?.visualKind}`, failures);
    record(inspection?.collisionMode === "unified-glb",
      `${objectId}: collisionMode=${inspection?.collisionMode}`, failures);
    record(inspection?.unifiedVisualCollision === true,
      `${objectId}: visual and collider do not share one PBR GLB`, failures);
    record(inspection?.pbrMaterialCount > 0,
      `${objectId}: pbrMaterialCount=${inspection?.pbrMaterialCount}`, failures);
    record(inspection?.bvhMeshCount > 0,
      `${objectId}: bvhMeshCount=${inspection?.bvhMeshCount}`, failures);
    record(inspection?.collisionFaces === definition?.collision?.asset?.faces,
      `${objectId}: collisionFaces=${inspection?.collisionFaces}`, failures);
  }

  const queries = [];
  queries.push(await askAndInspect(page, "左侧台灯在哪里？", "sam3_lamp_01"));
  screenshots.leftLampQa = await screenshot(page, outputDir, "desktop_qa_left_lamp_location");
  queries.push(await askAndInspect(page, "右侧台灯长什么样子？", "sam3_lamp_02"));
  screenshots.rightLampQa = await screenshot(page, outputDir, "desktop_qa_right_lamp_appearance");
  queries.push(await askAndInspect(page, "前排枕头在哪里？", "sam3_pillow_front"));
  screenshots.pillowLocationQa = await screenshot(page, outputDir, "desktop_qa_front_pillow_location");
  queries.push(await askAndInspect(page, "左侧后排枕头长什么样子？", "sam3_pillow_left"));
  screenshots.pillowAppearanceQa = await screenshot(page, outputDir, "desktop_qa_left_pillow_appearance");

  const expectedTerms = {
    sam3_lamp_01: ["左侧床头柜", "左侧"],
    sam3_lamp_02: ["灯罩", "陶瓷"],
    sam3_pillow_front: ["床垫", "床"],
    sam3_pillow_left: ["灰白", "米白", "柔软"],
  };
  for (const query of queries) {
    const answerText = String(query.answer?.answer || "");
    record(query.answer?.status === "resolved", `${query.question}: status=${query.answer?.status}`, failures);
    record(query.answer?.entityId === query.expectedId,
      `${query.question}: entity=${query.answer?.entityId}`, failures);
    record(query.answer?.focusEntityId === query.expectedId,
      `${query.question}: focus=${query.answer?.focusEntityId}`, failures);
    record(query.snapshot.state.selectedInteractiveObject === query.expectedId,
      `${query.question}: selected=${query.snapshot.state.selectedInteractiveObject}`, failures);
    record(query.snapshot.inspection?.selectionOutlineVisible === true,
      `${query.question}: bounding box is not visible`, failures);
    record(query.snapshot.focusVisibility?.passed === true,
      `${query.question}: focus visibility=${JSON.stringify(query.snapshot.focusVisibility)}`, failures);
    record(expectedTerms[query.expectedId].some((term) => answerText.includes(term)),
      `${query.question}: unexpected answer=${answerText}`, failures);
  }

  const drag = await testPointerDrag(page, "sam3_lamp_02");
  record(drag.target.point != null, "lamp02: no pointer hit target", failures);
  record(Number(drag.matrixDelta) > 1e-4, `lamp02: drag matrix delta=${drag.matrixDelta}`, failures);
  record(Number(drag.visualCollisionMatrixDelta) <= 1e-5,
    `lamp02: visual/collision matrix delta after drag=${drag.visualCollisionMatrixDelta}`, failures);
  const spin = await testSpin(page, "sam3_lamp_01");
  record(spin.start?.started === true, "lamp01: spin did not start", failures);
  record(Number(spin.duringMatrixDelta) > 1e-4,
    `lamp01: spin matrix did not change=${spin.duringMatrixDelta}`, failures);
  record(spin.completed === true, "lamp01: spin animation did not finish within 30 seconds", failures);
  record(spin.after.turns === spin.before.turns + 1,
    `lamp01: turns ${spin.before.turns} -> ${spin.after.turns}`, failures);
  record(Number(spin.returnMatrixDelta) <= 1e-5,
    `lamp01: full turn did not return=${spin.returnMatrixDelta}`, failures);
  record(Number(spin.visualCollisionMatrixDelta) <= 1e-5,
    `lamp01: visual/collision matrix delta=${spin.visualCollisionMatrixDelta}`, failures);
  screenshots.leftLampAfterSpin = await screenshot(page, outputDir, "desktop_left_lamp_after_spin");

  const finalDiagnostics = finalizeDiagnostics(diagnostics);
  record(finalDiagnostics.pageErrors.length === 0,
    `desktop: page errors=${JSON.stringify(finalDiagnostics.pageErrors)}`, failures);
  record(finalDiagnostics.consoleErrors.length === 0,
    `desktop: console errors=${JSON.stringify(finalDiagnostics.consoleErrors)}`, failures);
  record(finalDiagnostics.requestFailures.length === 0,
    `desktop: request failures=${JSON.stringify(finalDiagnostics.requestFailures)}`, failures);
  await page.close();
  return { status: failures.length ? "failed" : "passed", failures, diagnostics: finalDiagnostics, initial, canvas, layout, queries, drag, spin, screenshots };
}

async function runMobile(browser, url, outputDir, contract) {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  const diagnostics = listenForDiagnostics(page);
  const failures = [];
  console.log(JSON.stringify({ stage: "mobile_load", url }));
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
  await waitForReady(page, contract);
  const query = await askAndInspect(page, "右侧台灯在哪里？", "sam3_lamp_02");
  const canvas = await inspectCanvasPixels(page);
  const layout = await inspectLayout(page);
  const capture = await screenshot(page, outputDir, "mobile_qa_right_lamp_location");
  record(query.answer?.status === "resolved" && query.answer?.entityId === "sam3_lamp_02",
    `mobile: QA=${JSON.stringify(query.answer)}`, failures);
  record(query.snapshot.inspection?.selectionOutlineVisible === true,
    "mobile: selected lamp bounding box is not visible", failures);
  record(canvas.nonblank === true, `mobile: blank canvas=${JSON.stringify(canvas)}`, failures);
  record(layout.noHorizontalOverflow && layout.qaVisible && layout.canvasVisible,
    `mobile: invalid layout=${JSON.stringify(layout)}`, failures);
  const finalDiagnostics = finalizeDiagnostics(diagnostics);
  record(finalDiagnostics.pageErrors.length === 0 && finalDiagnostics.consoleErrors.length === 0
    && finalDiagnostics.requestFailures.length === 0,
  `mobile: browser diagnostics=${JSON.stringify(finalDiagnostics)}`, failures);
  await page.close();
  return { status: failures.length ? "failed" : "passed", failures, diagnostics: finalDiagnostics, query, canvas, layout, screenshot: capture };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const manifestPath = path.resolve(args.manifest || defaultManifest);
  const reportPath = path.resolve(args.report || defaultReport);
  const screenshotDir = path.resolve(args.screenshots || path.dirname(reportPath));
  const baseUrl = args.url || defaultUrl;
  const runtimeManifestUrl = args["runtime-manifest-url"] || null;
  const manifest = readJson(manifestPath);
  const contract = expectedContract(manifest);
  const failures = [];
  record(contract.objectCount === 10, `manifest object count=${contract.objectCount}`, failures);
  record(contract.staticVisualCount === 392593,
    `manifest static visual count=${contract.staticVisualCount}`, failures);
  record(contract.staticColliderFaces === 499999,
    `manifest static collider faces=${contract.staticColliderFaces}`, failures);
  record(PILLOW_IDS.every((id) => manifest.interactiveObjects.some((object) => object.id === id)),
    "manifest is missing reconstructed pillows", failures);
  record(LAMP_IDS.every((id) => manifest.interactiveObjects.some((object) => object.id === id)),
    "manifest is missing reconstructed lamps", failures);
  if (failures.length) throw new Error(failures.join("\n"));

  const url = runtimeUrl(baseUrl, manifestPath, runtimeManifestUrl);
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  let desktop;
  let mobile;
  try {
    desktop = await runDesktop(browser, url, screenshotDir, manifest, contract);
    mobile = await runMobile(browser, url, screenshotDir, contract);
  } finally {
    await browser.close();
  }
  const allFailures = [...desktop.failures, ...mobile.failures];
  const report = {
    schemaVersion: 1,
    kind: "video2world.bedroom4_lamp_replacement_browser_qa",
    createdAt: new Date().toISOString(),
    status: allFailures.length ? "failed" : "passed",
    failures: allFailures,
    runtimeUrl: url,
    manifest: fileDescriptor(manifestPath),
    contract,
    normalVisualMode: true,
    desktop,
    mobile,
  };
  writeJson(reportPath, report);
  console.log(JSON.stringify({ status: report.status, reportPath, failures: allFailures }, null, 2));
  if (allFailures.length) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
