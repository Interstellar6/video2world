import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { chromium } from "@playwright/test";

const CANONICAL_VIEWS = ["front", "right", "back", "left", "top", "bottom"];
const CANONICAL_VIEW_AXES = {
  front: { direction: [0, 0, 1], up: [0, 1, 0] },
  right: { direction: [1, 0, 0], up: [0, 1, 0] },
  back: { direction: [0, 0, -1], up: [0, 1, 0] },
  left: { direction: [-1, 0, 0], up: [0, 1, 0] },
  top: { direction: [0, 1, 0], up: [0, 0, -1] },
  bottom: { direction: [0, -1, 0], up: [0, 0, 1] },
};

function argument(name, fallback = null) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function writeJson(filePath, value) {
  const temporary = `${filePath}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`);
  fs.renameSync(temporary, filePath);
}

function decodePngDataUrl(value, viewId) {
  const match = /^data:image\/png;base64,([A-Za-z0-9+/=]+)$/u.exec(value || "");
  if (!match) throw new Error(`${viewId} render is not a base64 PNG data URL`);
  const bytes = Buffer.from(match[1], "base64");
  if (bytes.byteLength < 100) throw new Error(`${viewId} render PNG is unexpectedly small`);
  return bytes;
}

function sameOrderedValues(actual, expected) {
  return Array.isArray(actual)
    && actual.length === expected.length
    && actual.every((value, index) => value === expected[index]);
}

function canonicalAxesFromViewSpecs(viewSpecs) {
  if (!Array.isArray(viewSpecs) || viewSpecs.length !== CANONICAL_VIEWS.length) {
    throw new Error("Object review did not expose exactly six view specifications");
  }
  const axes = {};
  for (const [index, viewId] of CANONICAL_VIEWS.entries()) {
    const spec = viewSpecs[index];
    const expected = CANONICAL_VIEW_AXES[viewId];
    if (
      spec?.id !== viewId
      || !sameOrderedValues(spec.direction, expected.direction)
      || !sameOrderedValues(spec.up, expected.up)
    ) {
      throw new Error(`Object review axis mismatch for ${viewId}: ${JSON.stringify(spec)}`);
    }
    axes[viewId] = { direction: [...spec.direction], up: [...spec.up] };
  }
  return axes;
}

const url = argument("url");
const outputDir = path.resolve(argument("output-dir", "."));
const contactSheetName = argument("contact-sheet", "object_six_view_contact_sheet.png");
const receiptName = argument("receipt", "object_six_view_review.json");
if (!url) {
  throw new Error(
    "Usage: node scripts/capture_object_review.mjs --url <object-review-url> --output-dir <dir>",
  );
}
fs.mkdirSync(outputDir, { recursive: true });
const receiptPath = path.join(outputDir, receiptName);
const browser = await chromium.launch({ channel: "chrome", headless: true });
let receipt = {
  schemaVersion: "1.0",
  kind: "video2world.canonical_object_six_view_review",
  createdAt: new Date().toISOString(),
  state: "running",
  promotionAllowed: false,
  sourcePageUrl: url,
  canonicalViews: CANONICAL_VIEWS,
  canonicalViewAxes: CANONICAL_VIEW_AXES,
  orientationContract: "web/object-review.js VIEW_SPECS orthogonal object-local axes",
  horizontalOrbitAcceptedAsSixViewEvidence: false,
  visualGate: {
    status: "pending",
    blocking: [
      "shape_mismatch",
      "primary_color_category_error",
      "missing_surface_or_sheet_geometry",
      "disconnected_or_floating_parts",
      "significant_scene_interpenetration",
    ],
    warning: [
      "minor_backside_texture_hallucination",
      "minor_material_or_pattern_drift",
    ],
  },
};

try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1280 },
    deviceScaleFactor: 1,
  });
  const consoleWarnings = [];
  const consoleErrors = [];
  page.on("console", (message) => {
    if (message.type() === "warning") consoleWarnings.push(message.text());
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => consoleErrors.push(`pageerror: ${error.message}`));
  await page.route("**/favicon.ico", (route) => route.fulfill({ status: 204, body: "" }));
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => ["ready", "error"].includes(window.__VIDEO2WORLD_OBJECT_REVIEW__?.state),
    null,
    { timeout: 60_000 },
  );
  const state = await page.evaluate(() => structuredClone(window.__VIDEO2WORLD_OBJECT_REVIEW__));
  if (state.state !== "ready") throw new Error(`Object review failed: ${state.error}`);
  if (!sameOrderedValues(state.views, CANONICAL_VIEWS)) {
    throw new Error(`Object review views are not canonical: ${JSON.stringify(state.views)}`);
  }
  const pageViewAxes = canonicalAxesFromViewSpecs(state.viewSpecs);
  if (!state.renderDataUrls || Object.keys(state.renderDataUrls).length !== 6) {
    throw new Error("Object review did not expose exactly six object-only renders");
  }
  if (consoleErrors.length) {
    throw new Error(`Object review emitted browser errors: ${consoleErrors.join(" | ")}`);
  }

  const viewArtifacts = {};
  for (const viewId of CANONICAL_VIEWS) {
    const bytes = decodePngDataUrl(state.renderDataUrls[viewId], viewId);
    const fileName = `${viewId}_object.png`;
    const filePath = path.join(outputDir, fileName);
    fs.writeFileSync(filePath, bytes);
    viewArtifacts[viewId] = {
      file: fileName,
      path: filePath,
      sizeBytes: bytes.byteLength,
      sha256: sha256(bytes),
      ...pageViewAxes[viewId],
    };
  }

  const contactSheetPath = path.join(outputDir, contactSheetName);
  await page.locator("#review-grid").screenshot({ path: contactSheetPath });
  const contactSheetBytes = fs.readFileSync(contactSheetPath);
  const resolvedAssetUrl = new URL(state.assetUrl, page.url()).href;
  const assetResponse = await page.request.get(resolvedAssetUrl);
  if (!assetResponse.ok()) {
    throw new Error(`Unable to hash reviewed GLB: HTTP ${assetResponse.status()}`);
  }
  const assetBytes = await assetResponse.body();

  receipt = {
    ...receipt,
    state: "captured_visual_pending",
    canonicalViewAxes: pageViewAxes,
    objectId: state.objectId,
    renderMode: state.renderMode,
    bounds: state.bounds,
    meshCount: state.meshCount,
    triangleCount: state.triangleCount,
    asset: {
      url: resolvedAssetUrl,
      sizeBytes: assetBytes.byteLength,
      sha256: sha256(assetBytes),
    },
    views: viewArtifacts,
    contactSheet: {
      file: contactSheetName,
      path: contactSheetPath,
      sizeBytes: contactSheetBytes.byteLength,
      sha256: sha256(contactSheetBytes),
      layout: "3x2",
      viewport: [1440, 1280],
    },
    browser: {
      consoleWarnings,
      consoleErrors,
    },
  };
  writeJson(receiptPath, receipt);
  process.stdout.write(`${JSON.stringify({
    state: receipt.state,
    objectId: receipt.objectId,
    contactSheet: contactSheetPath,
    receipt: receiptPath,
  }, null, 2)}\n`);
} catch (error) {
  receipt = {
    ...receipt,
    state: "failed",
    error: error instanceof Error ? error.message : String(error),
  };
  writeJson(receiptPath, receipt);
  throw error;
} finally {
  await browser.close();
}
