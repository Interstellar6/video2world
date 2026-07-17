import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

const SH0_RENDER_MODE = "spark-sh0-direct-color";
const RENDER_WIDTH = 512;
const RENDER_HEIGHT = 384;
const VIEW_SPECS = [
  { id: "front", direction: [0, 0, 1], up: [0, 1, 0] },
  { id: "right", direction: [1, 0, 0], up: [0, 1, 0] },
  { id: "back", direction: [0, 0, -1], up: [0, 1, 0] },
  { id: "left", direction: [-1, 0, 0], up: [0, 1, 0] },
  { id: "top", direction: [0, 1, 0], up: [0, 0, -1] },
  { id: "bottom", direction: [0, -1, 0], up: [0, 0, 1] },
];

const params = new URLSearchParams(window.location.search);
const assetUrl = params.get("asset");
const objectId = params.get("objectId") || "unnamed-gaussian-object";
const title = document.querySelector("#object-title");
const grid = document.querySelector("#review-grid");
const status = document.querySelector("#review-status");
const stats = document.querySelector("#review-stats");
const renderStage = document.querySelector("#review-render-stage");

function parseVector(name, length) {
  const raw = params.get(name);
  if (raw == null || raw.trim() === "") return null;
  const values = raw.split(",").map((value) => Number(value.trim()));
  if (values.length !== length || values.some((value) => !Number.isFinite(value))) {
    throw new Error(`?${name}= must contain ${length} finite comma-separated numbers`);
  }
  return values;
}

function parseNumber(name, fallback) {
  const raw = params.get(name);
  if (raw == null || raw.trim() === "") return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`?${name}= must be a finite number`);
  return value;
}

function parseBoundsOverride() {
  const packed = parseVector("bbox", 6);
  const min = packed?.slice(0, 3) || parseVector("bboxMin", 3);
  const max = packed?.slice(3, 6) || parseVector("bboxMax", 3);
  if ((min == null) !== (max == null)) {
    throw new Error("Use both ?bboxMin= and ?bboxMax=, or one six-number ?bbox=");
  }
  if (min == null) return null;
  if (min.some((value, index) => value >= max[index])) {
    throw new Error("Explicit Gaussian review bbox must have min < max on every axis");
  }
  return new THREE.Box3(new THREE.Vector3(...min), new THREE.Vector3(...max));
}

const expectedExtents = parseVector("expectedExtents", 3);
const expectedMeanRgb = parseVector("expectedMeanRgb", 3);
const expectedColor = params.get("expectedColor") || null;
const reviewThresholds = {
  extentTolerance: parseNumber("extentTolerance", 0.12),
  minThicknessRatio: parseNumber("minThicknessRatio", 0.03),
  minLuminance: parseNumber("minLuminance", expectedColor ? 155 : 0),
  maxChroma: parseNumber("maxChroma", expectedColor ? 55 : 255),
  maxMeanRgbError: parseNumber("maxMeanRgbError", 55),
};

const reviewState = {
  schemaVersion: 1,
  kind: "video2world.gaussian_six_view_browser_review",
  state: "loading",
  objectId,
  assetUrl,
  assetSha256: null,
  assetBytes: 0,
  renderMode: SH0_RENDER_MODE,
  viewIds: VIEW_SPECS.map(({ id }) => id),
  pointCount: 0,
  sourceBounds: null,
  framingBounds: null,
  framingCenter: null,
  boundsOverride: false,
  sourceColor: null,
  expected: {
    extents: expectedExtents,
    meanRgb: expectedMeanRgb,
    colorFamily: expectedColor,
  },
  thresholds: reviewThresholds,
  viewMetrics: {},
  renderDataUrls: {},
  gates: null,
  blockers: [],
  promotionAllowed: false,
  error: null,
};
window.__VIDEO2WORLD_GAUSSIAN_REVIEW__ = reviewState;
title.textContent = objectId;

function formatVector(values) {
  return values.map((value) => Number(value).toFixed(4)).join(" x ");
}

function rounded(values, digits = 6) {
  return values.map((value) => Number(value.toFixed(digits)));
}

function addStat(label, value) {
  const item = document.createElement("div");
  item.className = "review-stat";
  const term = document.createElement("dt");
  term.textContent = label;
  const description = document.createElement("dd");
  description.textContent = value;
  item.append(term, description);
  stats.append(item);
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
}

function sourceColorStats(splat) {
  let count = 0;
  let red = 0;
  let green = 0;
  let blue = 0;
  splat.forEachSplat((_index, _center, _scales, _quaternion, opacity, color) => {
    if (!Number.isFinite(opacity) || opacity <= 0.01) return;
    count += 1;
    red += THREE.MathUtils.clamp(color.r, 0, 1) * 255;
    green += THREE.MathUtils.clamp(color.g, 0, 1) * 255;
    blue += THREE.MathUtils.clamp(color.b, 0, 1) * 255;
  });
  const meanRgb = count ? [red / count, green / count, blue / count] : [0, 0, 0];
  const luminance = 0.2126 * meanRgb[0] + 0.7152 * meanRgb[1] + 0.0722 * meanRgb[2];
  return {
    sampledPoints: count,
    meanRgb: rounded(meanRgb, 3),
    meanLuminance: Number(luminance.toFixed(3)),
    chroma: Number((Math.max(...meanRgb) - Math.min(...meanRgb)).toFixed(3)),
  };
}

function inspectPixels(renderer) {
  const gl = renderer.getContext();
  const pixels = new Uint8Array(RENDER_WIDTH * RENDER_HEIGHT * 4);
  gl.readPixels(0, 0, RENDER_WIDTH, RENDER_HEIGHT, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  let count = 0;
  let maxAlpha = 0;
  let minX = RENDER_WIDTH;
  let minY = RENDER_HEIGHT;
  let maxX = -1;
  let maxY = -1;
  let red = 0;
  let green = 0;
  let blue = 0;
  for (let index = 0; index < pixels.length; index += 4) {
    const alpha = pixels[index + 3];
    maxAlpha = Math.max(maxAlpha, alpha);
    if (alpha <= 8) continue;
    const pixelIndex = index / 4;
    const x = pixelIndex % RENDER_WIDTH;
    const y = Math.floor(pixelIndex / RENDER_WIDTH);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    const unpremultiply = alpha > 0 ? 255 / alpha : 0;
    red += Math.min(255, pixels[index] * unpremultiply);
    green += Math.min(255, pixels[index + 1] * unpremultiply);
    blue += Math.min(255, pixels[index + 2] * unpremultiply);
    count += 1;
  }
  const meanRgb = count ? [red / count, green / count, blue / count] : [0, 0, 0];
  const luminance = 0.2126 * meanRgb[0] + 0.7152 * meanRgb[1] + 0.0722 * meanRgb[2];
  const pixelBounds = count ? [minX, minY, maxX, maxY] : null;
  const margin = count ? Math.min(minX, minY, RENDER_WIDTH - 1 - maxX, RENDER_HEIGHT - 1 - maxY) : 0;
  return {
    canvasWidth: RENDER_WIDTH,
    canvasHeight: RENDER_HEIGHT,
    nonTransparentPixels: count,
    fillFraction: Number((count / (RENDER_WIDTH * RENDER_HEIGHT)).toFixed(6)),
    maxAlpha,
    pixelBounds,
    edgeMarginPixels: margin,
    meanRgb: rounded(meanRgb, 3),
    meanLuminance: Number(luminance.toFixed(3)),
    chroma: Number((Math.max(...meanRgb) - Math.min(...meanRgb)).toFixed(3)),
    nonblank: count >= 64 && maxAlpha > 8,
    framingOk: count >= 64 && margin >= 2,
  };
}

function makePanel(viewId) {
  const panel = document.createElement("article");
  panel.className = "review-view";
  panel.dataset.viewId = viewId;
  const label = document.createElement("p");
  label.className = "review-label";
  label.textContent = viewId;
  const canvas = document.createElement("canvas");
  canvas.width = RENDER_WIDTH;
  canvas.height = RENDER_HEIGHT;
  canvas.dataset.reviewCanvas = viewId;
  panel.append(label, canvas);
  grid.append(panel);
  return { panel, canvas };
}

async function drawDataUrl(canvas, dataUrl) {
  const image = new Image();
  image.src = dataUrl;
  await image.decode();
  const context = canvas.getContext("2d", { alpha: true });
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
}

async function renderView({ renderer, spark, scene, camera, center, radius, spec }) {
  const direction = new THREE.Vector3(...spec.direction);
  const up = new THREE.Vector3(...spec.up);
  const distance = Math.max(radius / Math.sin(THREE.MathUtils.degToRad(15)), 0.25) * 1.18;
  camera.position.copy(center).addScaledVector(direction, distance);
  camera.up.copy(up);
  camera.lookAt(center);
  camera.near = Math.max(distance / 100, 0.001);
  camera.far = Math.max(distance * 8, camera.near + 1);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  spark.sortDirty = true;
  spark.setDirty();
  let metrics = null;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const frameLimit = attempt === 0 ? 90 : 45;
    for (let frame = 0; frame < frameLimit; frame += 1) {
      renderer.clear(true, true, true);
      renderer.render(scene, camera);
      await nextFrame();
      if (frame >= 8 && !spark.sorting && spark.activeSplats > 0) break;
    }
    renderer.clear(true, true, true);
    renderer.render(scene, camera);
    await nextFrame();
    metrics = inspectPixels(renderer);
    if (metrics.nonblank) break;
    spark.sortDirty = true;
    spark.setDirty();
  }
  return { metrics, dataUrl: renderer.domElement.toDataURL("image/png") };
}

function buildGates(sourceBounds) {
  const size = sourceBounds.getSize(new THREE.Vector3()).toArray();
  const minExtent = Math.min(...size);
  const maxExtent = Math.max(...size);
  const thicknessRatio = maxExtent > 0 ? minExtent / maxExtent : 0;
  const viewValues = Object.values(reviewState.viewMetrics);
  const viewsNonblank = viewValues.length === VIEW_SPECS.length && viewValues.every((view) => view.nonblank);
  const framingCorrect = viewValues.length === VIEW_SPECS.length && viewValues.every((view) => view.framingOk);
  const sourceColor = reviewState.sourceColor;
  const renderedColorCorrect = !expectedColor || viewValues.every((view) => (
    view.meanLuminance >= reviewThresholds.minLuminance && view.chroma <= reviewThresholds.maxChroma
  ));
  const sourceColorCorrect = !expectedColor || (
    sourceColor.meanLuminance >= reviewThresholds.minLuminance
    && sourceColor.chroma <= reviewThresholds.maxChroma
  );
  const meanRgbCorrect = !expectedMeanRgb || sourceColor.meanRgb.every((value, index) => (
    Math.abs(value - expectedMeanRgb[index]) <= reviewThresholds.maxMeanRgbError
  ));
  const extentRelativeErrors = expectedExtents
    ? size.map((value, index) => Math.abs(value - expectedExtents[index]) / Math.max(Math.abs(expectedExtents[index]), 1e-9))
    : null;
  const extentsCorrect = !extentRelativeErrors
    || extentRelativeErrors.every((error) => error <= reviewThresholds.extentTolerance);
  const gates = {
    allSixViewsRendered: viewValues.length === VIEW_SPECS.length,
    allSixViewsNonblank: viewsNonblank,
    directColorMatchesExpectation: sourceColorCorrect && renderedColorCorrect && meanRgbCorrect,
    notSheetlike: thicknessRatio >= reviewThresholds.minThicknessRatio,
    boundsMatchExpectation: extentsCorrect,
    boundsFrameAllViews: framingCorrect,
  };
  reviewState.geometry = {
    extents: rounded(size),
    thicknessToMaxExtentRatio: Number(thicknessRatio.toFixed(6)),
    expectedExtentRelativeErrors: extentRelativeErrors ? rounded(extentRelativeErrors) : null,
  };
  reviewState.gates = gates;
  reviewState.blockers = Object.entries(gates)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
  reviewState.promotionAllowed = reviewState.blockers.length === 0;
}

async function loadReview() {
  if (!assetUrl) throw new Error("Missing required ?asset= URL parameter");
  const response = await fetch(assetUrl);
  if (!response.ok) throw new Error(`Gaussian asset request failed: HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  reviewState.assetBytes = bytes.byteLength;
  reviewState.assetSha256 = await sha256Hex(bytes);

  const renderer = new THREE.WebGLRenderer({
    antialias: false,
    alpha: true,
    preserveDrawingBuffer: true,
    premultipliedAlpha: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(1);
  renderer.setSize(RENDER_WIDTH, RENDER_HEIGHT, false);
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  renderStage.append(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, RENDER_WIDTH / RENDER_HEIGHT, 0.001, 1000);
  const spark = new SparkRenderer({ renderer, enableLod: false, sortRadial: false });
  scene.add(spark);
  const fileName = decodeURIComponent(new URL(assetUrl, window.location.href).pathname.split("/").pop() || "asset.ply");
  const splat = new SplatMesh({
    fileBytes: bytes,
    fileName,
    fileType: "ply",
    lod: false,
    raycastable: false,
  });
  splat.maxSh = 0;
  scene.add(splat);
  await splat.initialized;
  splat.maxSh = 0;
  splat.splats?.setMaxSh?.(0);
  splat.updateGenerator();

  reviewState.pointCount = splat.splats?.getNumSplats?.() || splat.packedSplats?.numSplats || 0;
  if (reviewState.pointCount <= 0) throw new Error("Loaded PLY contains no Gaussian splats");
  const sourceBounds = splat.getBoundingBox(true);
  if (sourceBounds.isEmpty()) throw new Error("Loaded Gaussian PLY has no finite center bounds");
  const boundsOverride = parseBoundsOverride();
  const framingBounds = boundsOverride || sourceBounds.clone();
  const centerOverride = parseVector("center", 3);
  const framingCenter = centerOverride
    ? new THREE.Vector3(...centerOverride)
    : framingBounds.getCenter(new THREE.Vector3());
  const sourceSize = sourceBounds.getSize(new THREE.Vector3());
  const framingSize = framingBounds.getSize(new THREE.Vector3());
  const radius = Math.max(framingSize.length() / 2, 1e-6);
  reviewState.sourceBounds = {
    min: rounded(sourceBounds.min.toArray()),
    max: rounded(sourceBounds.max.toArray()),
    size: rounded(sourceSize.toArray()),
  };
  reviewState.framingBounds = {
    min: rounded(framingBounds.min.toArray()),
    max: rounded(framingBounds.max.toArray()),
    size: rounded(framingSize.toArray()),
  };
  reviewState.framingCenter = rounded(framingCenter.toArray());
  reviewState.boundsOverride = Boolean(boundsOverride || centerOverride);
  reviewState.sourceColor = sourceColorStats(splat);

  addStat("Extents", formatVector(sourceSize.toArray()));
  addStat("Gaussians", reviewState.pointCount.toLocaleString("en-US"));
  addStat("Mean RGB", reviewState.sourceColor.meanRgb.map((value) => Math.round(value)).join(" / "));
  addStat("Mode", "SH0 direct");

  for (const spec of VIEW_SPECS) {
    const panel = makePanel(spec.id);
    const rendered = await renderView({ renderer, spark, scene, camera, center: framingCenter, radius, spec });
    reviewState.viewMetrics[spec.id] = rendered.metrics;
    reviewState.renderDataUrls[spec.id] = rendered.dataUrl;
    panel.panel.dataset.nonblank = String(rendered.metrics.nonblank);
    await drawDataUrl(panel.canvas, rendered.dataUrl);
    const artifact = document.createElement("img");
    artifact.className = "review-artifact";
    artifact.dataset.reviewRender = spec.id;
    artifact.alt = `${spec.id} Gaussian-only SH0 render`;
    artifact.src = rendered.dataUrl;
    document.body.append(artifact);
  }
  buildGates(sourceBounds);
  reviewState.state = "ready";
  status.dataset.state = reviewState.promotionAllowed ? "ready" : "rejected";
  status.textContent = reviewState.promotionAllowed
    ? "Gaussian six-view review passed all configured gates."
    : `Gaussian promotion blocked: ${reviewState.blockers.join(", ")}.`;
}

loadReview().catch((error) => {
  reviewState.state = "error";
  reviewState.error = error instanceof Error ? error.message : String(error);
  reviewState.promotionAllowed = false;
  status.textContent = `Review failed: ${reviewState.error}`;
  status.dataset.state = "error";
  console.error(error);
});
