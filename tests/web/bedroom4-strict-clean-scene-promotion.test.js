import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";

import { afterEach, describe, expect, test } from "vitest";

import {
  BEDROOM4_STABLE_ALIAS_SHA256,
  promoteBedroom4StrictCleanScene,
  recoverBedroom4StrictCleanSceneSwap,
} from "../../scripts/promote_bedroom4_strict_clean_scene.mjs";
import {
  APPROVED_NESTED_SUPPORT,
  deriveManifestExpectations,
} from "../../scripts/qa_bedroom4_clean_unified_scene_browser.mjs";
import { sha256, sha256File } from "../../scripts/lib/strict-clean-scene-assets.mjs";
import { validateWebManifest } from "../../web/web-manifest.js";

const roots = [];
const repoRoot = fileURLToPath(new URL("../..", import.meta.url));
const promotionScript = path.join(repoRoot, "scripts", "promote_bedroom4_strict_clean_scene.mjs");
const realStableAlias = path.join(
  repoRoot,
  "web",
  "public",
  "worlds",
  "bedroom4",
  "manifest.web-demo-baseline-stable.json",
);
const realStableWorld = path.dirname(realStableAlias);
const STABLE_PREFIX = "./worlds/bedroom4";
const IMPLEMENTATION_PATHS = {
  qaRunner: "scripts/qa_bedroom4_clean_unified_scene_browser.mjs",
  runtime: "web/visual-physics-proxy.js",
  manifestValidator: "web/web-manifest.js",
  materializer: "scripts/materialize_bedroom4_strict_clean_scene.mjs",
};
const LEGACY_IDS = [
  "sam3_nightstand_01",
  "sam3_nightstand_02",
  "sam3_plant_01",
  "sam3_plant_02",
];
const UNIFIED_IDS = [
  "sam3_pillow_front",
  "sam3_pillow_left",
  "sam3_pillow_right",
  "sam3_bed_01",
];
const ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE = "archived_current_demo_only";

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

function writeBytes(filePath, bytes) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, bytes);
}

function writeJson(filePath, value) {
  writeBytes(filePath, Buffer.from(`${JSON.stringify(value, null, 2)}\n`));
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

function linkOrCopy(source, target) {
  fs.mkdirSync(path.dirname(target), { recursive: true });
  if (fs.existsSync(target)) return;
  try {
    fs.linkSync(source, target);
  } catch {
    fs.copyFileSync(source, target);
  }
}

function installStableAliasClosure(targetWorld) {
  const manifest = JSON.parse(fs.readFileSync(realStableAlias, "utf8"));
  for (const url of new Set(collectPrefixUrls(manifest, STABLE_PREFIX))) {
    const relative = url.slice(`${STABLE_PREFIX}/`.length);
    linkOrCopy(path.join(realStableWorld, ...relative.split("/")), path.join(targetWorld, ...relative.split("/")));
  }
}

function installImplementationEvidence(projectRoot) {
  const evidence = {};
  for (const [key, relative] of Object.entries(IMPLEMENTATION_PATHS)) {
    const source = path.join(repoRoot, ...relative.split("/"));
    const target = path.join(projectRoot, ...relative.split("/"));
    linkOrCopy(source, target);
    evidence[key] = {
      path: `repo://video2world/${relative}`,
      size: fs.statSync(target).size,
      sha256: sha256File(target),
    };
  }
  return evidence;
}

let crc32Table = null;

function crc32(bytes) {
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

function pngChunk(type, data) {
  const typeBytes = Buffer.from(type, "ascii");
  const output = Buffer.alloc(12 + data.length);
  output.writeUInt32BE(data.length, 0);
  typeBytes.copy(output, 4);
  data.copy(output, 8);
  output.writeUInt32BE(crc32(Buffer.concat([typeBytes, data])), 8 + data.length);
  return output;
}

function asset(worldDir, prefix, relative, extra = {}) {
  const filePath = path.join(worldDir, ...relative.split("/"));
  writeBytes(filePath, Buffer.from(`fixture-${relative}`));
  return {
    url: `${prefix}/${relative}`,
    size: fs.statSync(filePath).size,
    sha256: sha256File(filePath),
    ...extra,
  };
}

function chunkedAsset(worldDir, prefix, relatives, extra = {}) {
  const buffers = relatives.map((relative) => {
    const filePath = path.join(worldDir, ...relative.split("/"));
    const bytes = Buffer.from(`fixture-${relative}`);
    writeBytes(filePath, bytes);
    return { relative, filePath, bytes };
  });
  const whole = Buffer.concat(buffers.map((item) => item.bytes));
  return {
    size: whole.length,
    sha256: sha256(whole),
    parts: buffers.map((item) => ({
      url: `${prefix}/${item.relative}`,
      size: item.bytes.length,
      sha256: sha256File(item.filePath),
    })),
    ...extra,
  };
}

function sceneAssets(worldDir, prefix) {
  return {
    visual: chunkedAsset(worldDir, prefix, [
      "chunks/static-visual.chunk000",
      "chunks/static-visual.chunk001",
    ], {
      id: "static_visual",
      fileName: "static.ply",
      fileType: "ply",
      format: "graphdeco-gaussian-ply",
      vertexCount: 100,
    }),
    collider: asset(worldDir, prefix, "chunks/raw-collider.chunk000", {
      id: "raw_collider",
      fileName: "raw.ply",
      fileType: "ply",
      format: "open3d-binary-little-endian-triangle-mesh-ply",
      faceCount: 50,
    }),
    colliderStaticCarved: asset(worldDir, prefix, "chunks/static-collider.chunk000", {
      id: "static_collider",
      fileName: "static-collider.ply",
      fileType: "ply",
      format: "open3d-binary-little-endian-triangle-mesh-ply",
      faceCount: 40,
    }),
  };
}

function legacyObject(worldDir, prefix, id, index) {
  return {
    id,
    placement: { pivot: [index, 0, 0], scale: [1, 1, 1], rotationEulerDeg: [0, 0, 0] },
    visual: asset(worldDir, prefix, `objects/${id}.visual.ply`, {
      fileName: `${id}.visual.ply`,
      fileType: "ply",
      format: "graphdeco-gaussian-ply",
      vertexCount: 10,
    }),
    colliderProxy: { type: "selection-box", center: [0, 0, 0], dimensions: [1, 1, 1] },
    collision: {
      mode: "kinematic",
      characterCollision: true,
      gate: { status: "passed" },
      asset: asset(worldDir, prefix, `objects/${id}.collider.glb`, {
        fileName: `${id}.collider.glb`,
        fileType: "glb",
        format: "gltf-binary",
        faces: 12,
      }),
    },
    interaction: { kind: "spin", degrees: 360, durationMs: 1000 },
    semanticGranularity: "independent_root_asset",
    parentObjectId: null,
    movesWithParent: false,
    independentlyMovable: true,
  };
}

function unifiedObject(worldDir, prefix, id, index) {
  const bed = id === "sam3_bed_01";
  return {
    id,
    placement: {
      pivot: [10 + index, 0, 0],
      scale: [1, 1, 1],
      rotationEulerDeg: [0, 0, 0],
      generatedCenter: [0, 0, 0],
    },
    collision: {
      mode: "unified-glb",
      topology: bed ? "closed_volume" : "surface_bvh",
      characterCollision: true,
      gate: {
        status: "passed",
        surfaceCollision: "passed",
        ...(bed ? {
          supportAdjustmentReceiptSha256: APPROVED_NESTED_SUPPORT.receiptSha256,
        } : {}),
      },
      asset: asset(worldDir, prefix, `objects/${id}.glb`, {
        fileName: `${id}.glb`,
        fileType: "glb",
        format: "gltf-binary",
        faces: 20 + index,
        vertices: 100 + index,
        finite: true,
        nondegenerate: true,
        windingConsistent: true,
        watertight: bed,
      }),
    },
    interaction: { kind: "spin", drag: "horizontal_yaw", degrees: 360, durationMs: 1000 },
    semanticGranularity: bed ? "independent_root_asset" : "independent_child_asset",
    parentObjectId: bed ? null : "sam3_bed_01",
    movesWithParent: !bed,
    childObjectIds: bed ? [...APPROVED_NESTED_SUPPORT.childObjectIds] : [],
    independentlyMovable: true,
  };
}

function minimalManifest({ worldDir, prefix, strictCandidate }) {
  const interactiveObjects = strictCandidate
    ? [
      ...LEGACY_IDS.map((id, index) => legacyObject(worldDir, prefix, id, index)),
      ...UNIFIED_IDS.map((id, index) => unifiedObject(worldDir, prefix, id, index)),
    ]
    : [];
  const manifest = {
    schemaVersion: 1,
    contract: "video2world-web-manifest-1.0.0",
    version: strictCandidate ? "strict-candidate-fixture" : "old-canonical-fixture",
    coordinateSystem: { worldUp: [0, -1, 0] },
    assets: sceneAssets(worldDir, prefix),
    interactiveObjects,
    sceneKnowledge: {
      objects: strictCandidate
        ? UNIFIED_IDS.map((id) => ({
          id,
          bbox: { min: [0, 0, 0], max: [1, 1, 1] },
          description: { short: { zh: `${id} fixture` } },
        }))
        : [],
    },
    collisionWorld: { sceneAssetKey: "colliderStaticCarved" },
    sourceWorld: {
      worldId: "bedroom_4",
      runId: "fixture",
      adoptionMode: "fixture",
    },
  };
  if (strictCandidate) {
    manifest.initialState = { cameraFocusObjectId: "sam3_bed_01" };
    manifest.candidateBuild = {
      status: "candidate_materialized_strict_clean_scene_browser_qa_pending",
      lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
      correctedFullPipeline: false,
      canonicalLayeredCompletion: false,
      promotionAllowed: false,
      promotionBlockers: ["candidate_browser_qa_pending"],
      report: `${prefix}/qa/strict-clean-scene-adoption-report.json`,
      cleanScene: {
        status: "strict_layered_clean_scene_materialized_current_demo_only",
        lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
        correctedFullPipeline: false,
        acceptanceScope: "current_demo_only",
        promotionApproved: false,
      },
      unifiedPbrObjects: { objectIds: [...UNIFIED_IDS] },
    };
    manifest.productionBuild = {
      strictCleanSceneCandidate: { status: "materialized_browser_qa_pending", promotionApproved: false },
    };
  }
  validateWebManifest(manifest);
  return manifest;
}

function fakePng(width, height) {
  const signature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 0;
  const row = Buffer.alloc(width + 1, 0x7f);
  row[0] = 0;
  const pixels = Buffer.concat(Array.from({ length: height }, () => row));
  return Buffer.concat([
    signature,
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", zlib.deflateSync(pixels)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}

function screenshotDescriptor(filePath, width, height) {
  return {
    path: filePath,
    sha256: sha256File(filePath),
    size: fs.statSync(filePath).size,
    width,
    height,
  };
}

function motionMetric({ moved, translated = false }) {
  return {
    matrixDelta: moved ? 1 : 0,
    translationDelta: translated ? 1 : 0,
    gramDelta: 0,
  };
}

function transformMotion({ moved, translated = false }) {
  return {
    groupMatrixWorld: motionMetric({ moved, translated }),
    splatMatrixWorld: motionMetric({ moved, translated }),
    collisionMatrixWorld: motionMetric({ moved, translated }),
  };
}

function hierarchyEvidence() {
  const pillows = UNIFIED_IDS.slice(0, 3);
  return {
    attachments: Object.fromEntries(pillows.map((id) => [id, 0])),
    initialMotionRootGramDelta: Object.fromEntries(pillows.map((id) => [id, 0])),
    parentMotion: Object.fromEntries(UNIFIED_IDS.map((id) => [id, {
      ...transformMotion({ moved: true, translated: id !== "sam3_bed_01" }),
      ...(id === "sam3_bed_01" ? {} : { relativeParentMatrixDelta: 0 }),
    }])),
    parentReturned: Object.fromEntries(UNIFIED_IDS.map((id) => [id, {
      group: true,
      visual: true,
      collision: true,
    }])),
    independentChildren: pillows.map((objectId) => ({
      objectId,
      pillowMotion: transformMotion({ moved: true }),
      bedMotion: transformMotion({ moved: false }),
      returned: { group: true, visual: true, collision: true },
    })),
  };
}

function runtimeEvidence(manifest, expected) {
  const interactiveObjects = manifest.interactiveObjects.map((object) => ({
    id: object.id,
    ready: true,
    visualReady: true,
    colliderReady: true,
    collisionMode: object.collision.mode,
  }));
  const collisionObjects = manifest.interactiveObjects.map((object) => ({
    id: object.id,
    ready: true,
    faces: object.collision.asset.faces,
    topology: object.collision.topology,
  }));
  const inspections = Object.fromEntries(UNIFIED_IDS.map((id) => {
    const object = manifest.interactiveObjects.find((item) => item.id === id);
    return [id, {
      visualReady: true,
      colliderReady: true,
      visualKind: "mesh",
      collisionMode: "unified-glb",
      collisionTopology: object.collision.topology,
      unifiedVisualCollision: true,
      collisionFaces: object.collision.asset.faces,
      meshVertexCount: object.collision.asset.vertices,
      pbrMaterialCount: 1,
      bvhMeshCount: 1,
      proxyWorldBounds: null,
      proxyMatrixWorld: null,
    }];
  }));
  return {
    initialState: {
      visualReady: true,
      colliderReady: true,
      sceneQaReady: true,
      cameraOverviewFocusObjectId: expected.cameraFocusObjectId,
      cameraInsideInteractiveObjectIds: [],
      cameraOverviewCoverage: {
        widthFraction: 0.5,
        heightFraction: 0.4,
      },
      staticVisualCount: expected.staticVisualCount,
      colliderFaces: expected.staticColliderFaces,
      interactiveObjectCount: expected.objectCount,
      interactiveObjectReadyCount: expected.objectCount,
      objectColliderCount: expected.objectColliderCount,
      interactiveObjectMeshVertexCount: expected.meshVertices,
      interactiveObjectGaussianCount: expected.gaussianVertices,
      interactiveObjectRgbPointCount: expected.rgbPointVertices,
      visualCount: expected.totalVisualPrimitives,
      interactiveObjects,
    },
    collisionWorld: {
      sceneReady: true,
      objectColliderReadyCount: expected.objectColliderCount,
      objectColliderFaces: expected.objectColliderFaces,
      degradedCount: 0,
      errors: [],
      objects: collisionObjects,
    },
    inspections,
  };
}

function queryEvidence(queryCase) {
  return {
    ...queryCase,
    answer: {
      status: "resolved",
      entityId: queryCase.expectedId,
      focusEntityId: queryCase.expectedId,
      answer: `${queryCase.expectedId} fixture answer`,
    },
    focusAndBbox: {
      selectedSceneEntity: queryCase.expectedId,
      selectedInteractiveObject: queryCase.expectedId,
      collisionBounds: { min: [0, 0, 0], max: [1, 1, 1] },
      selectionOutlineVisible: true,
      cameraMaximumDelta: 1,
      cameraTargetBoundsEvidence: {
        passed: true,
        centerDistance: 0,
        normalizedCenterDistance: 0,
        targetInsideBounds: true,
      },
    },
  };
}

function interactionEvidence(objectId) {
  return {
    objectId,
    focus: {
      focused: true,
      selectedSceneEntity: objectId,
      selectedInteractiveObject: objectId,
      selectionOutlineVisible: true,
    },
    pointer: { hit: { objectId, colliderMode: "unified-glb" } },
    drag: {
      deltas: { group: 1, visual: 1, collision: 1 },
      returned: { group: true, visual: true, collision: true },
    },
    doubleClick360: {
      turnsBefore: 0,
      turnsAfter: 1,
      inFlightDeltas: { visual: 1, collision: 1 },
      returned: { group: true, visual: true, collision: true },
    },
  };
}

function viewportEvidence({
  label,
  viewport,
  full,
  manifest,
  expected,
  manifestSha,
  runtimeUrl,
  screenshot,
}) {
  const queries = (full
    ? expected.queryCases
    : expected.queryCases.filter((item) => ["pillow_appearance", "bed_appearance"].includes(item.label)))
    .map(queryEvidence);
  const assetRequestCounts = Object.fromEntries(
    Object.values(expected.unified).map((contract) => [new URL(contract.url, runtimeUrl).pathname, 1]),
  );
  return {
    label,
    viewport,
    freshPageLoad: true,
    runtime: runtimeEvidence(manifest, expected),
    queries,
    interactions: full ? UNIFIED_IDS.map(interactionEvidence) : [],
    hierarchy: full ? hierarchyEvidence() : null,
    robotCollisions: full ? UNIFIED_IDS.map((objectId) => ({
      objectId,
      passed: true,
      prepared: { prepared: true },
      step: { blocked: true, lastCollisionObjectId: objectId },
    })) : [],
    interpenetrations: full ? {
      automatedStatus: "passed_with_recorded_contacts",
      classifications: [{ blocking: false }],
    } : null,
    canvas: {
      nonblank: true,
      nonDarkPixels: 100,
      quantizedColorCount: 16,
      lumaRange: 20,
    },
    layout: {
      viewport,
      checks: {
        noHorizontalOverflow: true,
        queryBeforeAnswer: true,
        answerBeforeMetrics: true,
        statusBeforeControls: true,
        answerNoOwnOverflow: true,
        exactViewport: true,
        exactCanvasCssSize: true,
      },
    },
    performance: {
      samples: [30, 30, 30, 30, 30],
      average: 30,
      minimum: 30,
      threshold: 18,
      passed: true,
    },
    diagnostics: {
      label,
      consoleMessages: [],
      pageErrors: [],
      requestFailures: [],
      httpErrors: [],
      manifestRequestCount: 1,
      manifestResponseSha256: [manifestSha],
      assetRequestCounts,
    },
    screenshot,
  };
}

function browserQaReport({ manifest, manifestPath, manifestSha, candidateWorld, implementation }) {
  const expected = deriveManifestExpectations(manifest);
  const runtimeUrl = new URL("http://127.0.0.1:4177/");
  runtimeUrl.searchParams.set("manifest", "./worlds/bedroom4-strict-clean-candidate/manifest.json");
  runtimeUrl.searchParams.set("collisionDebug", "on");
  const desktopScreenshot = path.join(candidateWorld, "qa", "screens", "desktop-1440x900.png");
  const mobileScreenshot = path.join(candidateWorld, "qa", "screens", "mobile-390x844.png");
  writeBytes(desktopScreenshot, fakePng(1440, 900));
  writeBytes(mobileScreenshot, fakePng(390, 844));
  const now = new Date().toISOString();
  return {
    schemaVersion: 2,
    kind: "video2world.bedroom4_clean_unified_scene_browser_qa",
    status: "passed_current_demo_only",
    automatedGate: "passed",
    acceptanceScope: "current_demo_only",
    promotionApproved: false,
    publishingPerformed: false,
    manifestEdited: false,
    startedAt: now,
    finishedAt: now,
    durationSeconds: 0,
    implementation,
    url: runtimeUrl.toString(),
    manifest: {
      path: manifestPath,
      sha256Before: manifestSha,
      sha256After: manifestSha,
      unchanged: true,
    },
    expected,
    qualityPolicy: {
      minorLimitationsMayPass: true,
      supportAndInterpenetrationAlwaysRecorded: true,
      obviousInterpenetrationBlocks: true,
      humanScreenshotReviewRequired: true,
    },
    acceptedLimitations: [],
    desktop: viewportEvidence({
      label: "desktop",
      viewport: [1440, 900],
      full: true,
      manifest,
      expected,
      manifestSha,
      runtimeUrl,
      screenshot: screenshotDescriptor(desktopScreenshot, 1440, 900),
    }),
    mobile: viewportEvidence({
      label: "mobile",
      viewport: [390, 844],
      full: false,
      manifest,
      expected,
      manifestSha,
      runtimeUrl,
      screenshot: screenshotDescriptor(mobileScreenshot, 390, 844),
    }),
    failures: [],
  };
}

function makeFixture() {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "video2world-swap-")));
  roots.push(root);
  const projectRoot = path.join(root, "project");
  const worldsRoot = path.join(projectRoot, "web", "public", "worlds");
  const canonicalWorld = path.join(worldsRoot, "bedroom4");
  const candidateWorld = path.join(worldsRoot, "bedroom4-strict-clean-candidate");
  const rollbackRoot = path.join(root, "project", "rollback");
  const backupWorld = path.join(rollbackRoot, "bedroom4-before-strict-clean");
  fs.mkdirSync(canonicalWorld, { recursive: true });
  fs.mkdirSync(candidateWorld, { recursive: true });
  fs.mkdirSync(rollbackRoot, { recursive: true });
  const implementation = installImplementationEvidence(projectRoot);
  const canonicalPrefix = "./worlds/bedroom4";
  const candidatePrefix = "./worlds/bedroom4-strict-clean-candidate";

  const canonicalManifest = minimalManifest({
    worldDir: canonicalWorld,
    prefix: canonicalPrefix,
    strictCandidate: false,
  });
  const publicManifest = path.join(canonicalWorld, "manifest.json");
  writeJson(publicManifest, canonicalManifest);
  const canonicalManifestSha = sha256File(publicManifest);
  writeBytes(path.join(canonicalWorld, "chunks", "legacy-old.chunk000"), Buffer.from("old-chunk"));
  const stableAlias = path.join(canonicalWorld, "manifest.web-demo-baseline-stable.json");
  fs.copyFileSync(realStableAlias, stableAlias);
  installStableAliasClosure(canonicalWorld);
  const stableAliasSha = sha256File(stableAlias);
  expect(stableAliasSha).toBe(BEDROOM4_STABLE_ALIAS_SHA256);

  const candidateManifest = minimalManifest({
    worldDir: candidateWorld,
    prefix: candidatePrefix,
    strictCandidate: true,
  });
  fs.copyFileSync(stableAlias, path.join(candidateWorld, path.basename(stableAlias)));
  installStableAliasClosure(candidateWorld);
  writeBytes(path.join(candidateWorld, "chunks", "legacy-old.chunk000"), Buffer.from("old-chunk"));
  const candidateManifestPath = path.join(candidateWorld, "manifest.json");
  writeJson(candidateManifestPath, candidateManifest);
  const candidateManifestSha = sha256File(candidateManifestPath);

  const adoptionReport = {
    schemaVersion: 1,
    kind: "video2world.strict_clean_scene_adoption_report",
    status: "candidate_materialized_browser_qa_pending",
    acceptanceScope: "current_demo_only",
    lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
    correctedFullPipeline: false,
    promotionApproved: false,
    browserQa: "pending",
    outputManifest: { sha256: candidateManifestSha },
    publicManifest: {
      sha256Before: canonicalManifestSha,
      sha256After: canonicalManifestSha,
      unchanged: true,
    },
    runtimeAssetResolution: {
      status: "passed",
      allManifestWorldUrlsResolve: true,
      runtimeAssetHashesRevalidated: true,
    },
    gates: { fixtureGate: true },
  };
  const adoptionReportPath = path.join(candidateWorld, "qa", "strict-clean-scene-adoption-report.json");
  writeJson(adoptionReportPath, adoptionReport);
  const candidateBrowserQaReport = path.join(candidateWorld, "qa", "candidate-browser-qa.json");
  writeJson(candidateBrowserQaReport, browserQaReport({
    manifest: candidateManifest,
    manifestPath: candidateManifestPath,
    manifestSha: candidateManifestSha,
    candidateWorld,
    implementation,
  }));
  return {
    root,
    canonicalWorld,
    candidateWorld,
    backupWorld,
    publicManifest,
    canonicalManifestSha,
    candidateManifestPath,
    candidateManifestSha,
    candidateBrowserQaReport,
    stableAliasSha,
    candidatePrefix,
    canonicalPrefix,
    implementation,
  };
}

function rebindCandidateEvidence(fixture, manifest) {
  validateWebManifest(manifest);
  writeJson(fixture.candidateManifestPath, manifest);
  fixture.candidateManifestSha = sha256File(fixture.candidateManifestPath);
  const adoptionPath = path.join(
    fixture.candidateWorld,
    "qa",
    "strict-clean-scene-adoption-report.json",
  );
  const adoption = JSON.parse(fs.readFileSync(adoptionPath));
  adoption.outputManifest.sha256 = fixture.candidateManifestSha;
  writeJson(adoptionPath, adoption);
  writeJson(fixture.candidateBrowserQaReport, browserQaReport({
    manifest,
    manifestPath: fixture.candidateManifestPath,
    manifestSha: fixture.candidateManifestSha,
    candidateWorld: fixture.candidateWorld,
    implementation: fixture.implementation,
  }));
}

function promote(fixture, overrides = {}) {
  return promoteBedroom4StrictCleanScene({
    candidateWorld: fixture.candidateWorld,
    candidateBrowserQaReport: fixture.candidateBrowserQaReport,
    canonicalSourceWorld: fixture.canonicalWorld,
    backupWorld: fixture.backupWorld,
    publicManifest: fixture.publicManifest,
    ...overrides,
  });
}

function collectStrings(value, output = []) {
  if (typeof value === "string") output.push(value);
  else if (Array.isArray(value)) value.forEach((item) => collectStrings(item, output));
  else if (value && typeof value === "object") Object.values(value).forEach((item) => collectStrings(item, output));
  return output;
}

describe.skipIf(!fs.existsSync(realStableAlias))(
  "strict clean-scene candidate promotion swap",
  () => {
  test("journal-swaps worlds, rewrites the prefix, and preserves rollback assets", () => {
    const fixture = makeFixture();
    const result = promote(fixture);

    expect(fs.existsSync(fixture.candidateWorld)).toBe(false);
    expect(fs.existsSync(fixture.canonicalWorld)).toBe(true);
    expect(fs.existsSync(fixture.backupWorld)).toBe(true);
    expect(sha256File(path.join(fixture.backupWorld, "manifest.json")))
      .toBe(fixture.canonicalManifestSha);
    expect(fs.readFileSync(path.join(fixture.backupWorld, "chunks", "legacy-old.chunk000"), "utf8"))
      .toBe("old-chunk");
    expect(sha256File(path.join(fixture.backupWorld, "manifest.web-demo-baseline-stable.json")))
      .toBe(fixture.stableAliasSha);
    expect(sha256File(path.join(fixture.canonicalWorld, "manifest.web-demo-baseline-stable.json")))
      .toBe(fixture.stableAliasSha);

    const manifest = JSON.parse(fs.readFileSync(fixture.publicManifest));
    expect(() => validateWebManifest(manifest)).not.toThrow();
    expect(manifest.candidateBuild.status).toBe("materialized_pending_promoted_manifest_recheck");
    expect(manifest.candidateBuild.promotionAllowed).toBe(false);
    expect(manifest.candidateBuild.lineageScope).toBe(ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE);
    expect(manifest.candidateBuild.correctedFullPipeline).toBe(false);
    expect(manifest.candidateBuild.canonicalLayeredCompletion).toBe(false);
    expect(manifest.candidateBuild.promotionSwap.finalQaClaimed).toBe(false);
    expect(manifest.candidateBuild.promotionSwap).toMatchObject({
      lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
      correctedFullPipeline: false,
    });
    expect(manifest.sourceWorld).toMatchObject({
      lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
      correctedFullPipeline: false,
    });
    const strings = collectStrings(manifest);
    expect(strings.some((value) => value.startsWith(`${fixture.candidatePrefix}/`))).toBe(false);
    expect(strings.some((value) => value.startsWith(`${fixture.canonicalPrefix}/`))).toBe(true);

    const receipt = JSON.parse(fs.readFileSync(result.receiptPath));
    expect(receipt).toMatchObject({
      status: "materialized_pending_promoted_manifest_recheck",
      lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
      correctedFullPipeline: false,
      promotionAllowed: false,
      promotionApproved: false,
      finalQaClaimed: false,
    });
    expect(receipt.promotedManifest).toMatchObject({
      sha256: result.promotedManifestSha256,
      browserQaRecheck: "required_pending",
    });
    expect(receipt.rollback).toMatchObject({
      automaticOnFailure: true,
      canonicalRestoreFrom: fixture.backupWorld,
      candidateRestoreTo: fixture.candidateWorld,
      oldCanonicalManifestSha256: fixture.canonicalManifestSha,
      candidateManifestSha256BeforeRewrite: fixture.candidateManifestSha,
    });
    expect(receipt.gates).toMatchObject({
      desktopAndMobileRuntimeQaRevalidated: true,
      manifestAssetDescriptorsRehashed: true,
      localAssetUrlsBrowserNormalizedAndBounded: true,
      realpathAndSymlinkBoundariesPassed: true,
      candidatePrefixRewrittenToCanonical: true,
      oldCanonicalWorldPreservedAtBackup: true,
      oldChunksPreservedAtBackup: true,
      promotedManifestRecheckRequired: true,
      finalQaNotClaimed: true,
    });
    const journal = JSON.parse(fs.readFileSync(result.journalPath));
    expect(journal).toMatchObject({
      status: "complete_pending_promoted_manifest_recheck",
      phase: "complete",
      lineageScope: ARCHIVED_CURRENT_DEMO_LINEAGE_SCOPE,
      correctedFullPipeline: false,
      promotionAllowed: false,
    });
    expect(receipt.swap.mode).toBe("durable_journaled_two_rename_transaction");
  });

  test("rejects a tampered QA report before touching either world", () => {
    const fixture = makeFixture();
    const qa = JSON.parse(fs.readFileSync(fixture.candidateBrowserQaReport));
    qa.failures = ["tampered blocking failure"];
    writeJson(fixture.candidateBrowserQaReport, qa);

    expect(() => promote(fixture)).toThrow(/QA failures are not empty/u);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
    expect(sha256File(fixture.candidateManifestPath)).toBe(fixture.candidateManifestSha);
    expect(fs.existsSync(fixture.backupWorld)).toBe(false);
  });

  test("rejects a nonempty backup target", () => {
    const fixture = makeFixture();
    writeBytes(path.join(fixture.backupWorld, "do-not-overwrite.txt"), Buffer.from("keep"));

    expect(() => promote(fixture)).toThrow(/backup-world must not exist/u);
    expect(fs.readFileSync(path.join(fixture.backupWorld, "do-not-overwrite.txt"), "utf8"))
      .toBe("keep");
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
    expect(sha256File(fixture.candidateManifestPath)).toBe(fixture.candidateManifestSha);
  });

  test("rolls back the first rename and restores the original candidate manifest", () => {
    const fixture = makeFixture();
    let calls = 0;
    const renameDirectory = (source, target) => {
      calls += 1;
      if (calls === 2) throw new Error("injected second rename failure");
      fs.renameSync(source, target);
    };

    expect(() => promote(fixture, { renameDirectory }))
      .toThrow(/injected second rename failure/u);
    expect(calls).toBe(3);
    expect(fs.existsSync(fixture.canonicalWorld)).toBe(true);
    expect(fs.existsSync(fixture.candidateWorld)).toBe(true);
    expect(fs.existsSync(fixture.backupWorld)).toBe(false);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
    expect(sha256File(fixture.candidateManifestPath)).toBe(fixture.candidateManifestSha);
    const journalPath = `${fixture.backupWorld}.swap-journal.json`;
    expect(JSON.parse(fs.readFileSync(journalPath))).toMatchObject({
      status: "recovered_rolled_back",
      phase: "recovered_rolled_back",
    });
  });

  test("rejects asset byte tampering even when manifest and QA hashes remain bound", () => {
    const fixture = makeFixture();
    const manifest = JSON.parse(fs.readFileSync(fixture.candidateManifestPath));
    const visualUrl = manifest.assets.visual.parts[0].url;
    const visualPath = path.join(
      fixture.candidateWorld,
      ...visualUrl.slice(`${fixture.candidatePrefix}/`.length).split("/"),
    );
    fs.appendFileSync(visualPath, "tamper");

    expect(() => promote(fixture)).toThrow(/assets\.visual\.parts\[0\] size mismatch/u);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
    expect(fs.existsSync(fixture.backupWorld)).toBe(false);
  });

  test("reconstructs chunked whole-asset SHA after every part hash passes", () => {
    const fixture = makeFixture();
    const manifest = JSON.parse(fs.readFileSync(fixture.candidateManifestPath));
    const part = manifest.assets.visual.parts[0];
    const partPath = path.join(
      fixture.candidateWorld,
      ...part.url.slice(`${fixture.candidatePrefix}/`.length).split("/"),
    );
    const bytes = fs.readFileSync(partPath);
    writeBytes(partPath, Buffer.alloc(bytes.length, 0x31));
    part.sha256 = sha256File(partPath);
    rebindCandidateEvidence(fixture, manifest);

    expect(() => promote(fixture)).toThrow(/assets\.visual reconstructed SHA-256 mismatch/u);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
  });

  test.each([
    "chunks/static-visual.chunk000?cache=1",
    "chunks/static-visual.chunk000#fragment",
    "chunks/%2e%2e/static-visual.chunk000",
    "chunks/../static-visual.chunk000",
  ])("rejects browser-ambiguous candidate asset URL %s", (relative) => {
    const fixture = makeFixture();
    const manifest = JSON.parse(fs.readFileSync(fixture.candidateManifestPath));
    manifest.assets.visual.parts[0].url = `${fixture.candidatePrefix}/${relative}`;
    rebindCandidateEvidence(fixture, manifest);

    expect(() => promote(fixture)).toThrow(/unsafe URL|normalization/u);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
  });

  test("rejects a candidate path with a symlink ancestor", () => {
    const fixture = makeFixture();
    const linkedCandidate = path.join(fixture.root, "candidate-link");
    fs.symlinkSync(fixture.candidateWorld, linkedCandidate, "dir");

    expect(() => promote(fixture, { candidateWorld: linkedCandidate }))
      .toThrow(/symlink ancestor/u);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
  });

  test("rejects forged passed status when desktop runtime evidence fails", () => {
    const fixture = makeFixture();
    const qa = JSON.parse(fs.readFileSync(fixture.candidateBrowserQaReport));
    qa.desktop.performance.average = 1;
    writeJson(fixture.candidateBrowserQaReport, qa);

    expect(() => promote(fixture)).toThrow(/FPS evidence failed/u);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
  });

  test.each([
    [
      "a different overview focus",
      (state) => { state.cameraOverviewFocusObjectId = "sam3_pillow_front"; },
      /camera overview focus mismatch/u,
    ],
    [
      "a camera inside an interactive object",
      (state) => { state.cameraInsideInteractiveObjectIds = ["sam3_bed_01"]; },
      /camera is inside an interactive object/u,
    ],
    [
      "a non-finite overview width",
      (state) => { state.cameraOverviewCoverage.widthFraction = null; },
      /cameraOverviewCoverage\.widthFraction must be finite/u,
    ],
    [
      "an undersized overview width",
      (state) => { state.cameraOverviewCoverage.widthFraction = 0.0799; },
      /camera overview width coverage out of range/u,
    ],
    [
      "an oversized overview height",
      (state) => { state.cameraOverviewCoverage.heightFraction = 0.9201; },
      /camera overview height coverage out of range/u,
    ],
  ])("rejects forged passed QA with %s", (_label, mutate, message) => {
    const fixture = makeFixture();
    const qa = JSON.parse(fs.readFileSync(fixture.candidateBrowserQaReport));
    mutate(qa.desktop.runtime.initialState);
    writeJson(fixture.candidateBrowserQaReport, qa);

    expect(() => promote(fixture)).toThrow(message);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
    expect(fs.existsSync(fixture.backupWorld)).toBe(false);
  });

  test("rejects forged passed QA whose query target does not frame the selected object", () => {
    const fixture = makeFixture();
    const qa = JSON.parse(fs.readFileSync(fixture.candidateBrowserQaReport));
    qa.desktop.queries[0].focusAndBbox.cameraMaximumDelta = 0;
    qa.desktop.queries[0].focusAndBbox.cameraTargetBoundsEvidence = {
      passed: false,
      centerDistance: 100,
      normalizedCenterDistance: 2,
      targetInsideBounds: false,
    };
    writeJson(fixture.candidateBrowserQaReport, qa);

    expect(() => promote(fixture)).toThrow(/bbox\/camera focus evidence failed/u);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
    expect(fs.existsSync(fixture.backupWorld)).toBe(false);
  });

  test("requires QA report location and bound manifest path inside the candidate world", () => {
    const fixture = makeFixture();
    const outsideQa = path.join(fixture.root, "outside-qa.json");
    fs.copyFileSync(fixture.candidateBrowserQaReport, outsideQa);
    expect(() => promote(fixture, { candidateBrowserQaReport: outsideQa }))
      .toThrow(/must be inside candidate-world\/qa/u);

    const qa = JSON.parse(fs.readFileSync(fixture.candidateBrowserQaReport));
    qa.manifest.path = fixture.publicManifest;
    writeJson(fixture.candidateBrowserQaReport, qa);
    expect(() => promote(fixture)).toThrow(/binds another manifest path/u);
  });

  test("does not allow the production stable alias pin to be overridden through the API", () => {
    const fixture = makeFixture();
    const candidateStable = path.join(
      fixture.candidateWorld,
      "manifest.web-demo-baseline-stable.json",
    );
    writeJson(candidateStable, { forged: true });
    const forgedSha = sha256File(candidateStable);

    expect(() => promote(fixture, { expectedStableAliasSha256: forgedSha }))
      .toThrow(/candidate stable alias SHA-256 mismatch/u);
  });

  test("enforces pillow children of the bed before any transaction state is written", () => {
    const fixture = makeFixture();
    const manifest = JSON.parse(fs.readFileSync(fixture.candidateManifestPath));
    const pillow = manifest.interactiveObjects.find((item) => item.id === "sam3_pillow_left");
    pillow.semanticGranularity = "independent_root_asset";
    pillow.parentObjectId = null;
    pillow.movesWithParent = false;
    writeJson(fixture.candidateManifestPath, manifest);
    fixture.candidateManifestSha = sha256File(fixture.candidateManifestPath);
    const adoptionPath = path.join(
      fixture.candidateWorld,
      "qa",
      "strict-clean-scene-adoption-report.json",
    );
    const adoption = JSON.parse(fs.readFileSync(adoptionPath));
    adoption.outputManifest.sha256 = fixture.candidateManifestSha;
    writeJson(adoptionPath, adoption);
    const qa = JSON.parse(fs.readFileSync(fixture.candidateBrowserQaReport));
    qa.manifest.sha256Before = fixture.candidateManifestSha;
    qa.manifest.sha256After = fixture.candidateManifestSha;
    writeJson(fixture.candidateBrowserQaReport, qa);

    expect(() => promote(fixture)).toThrow(/must be an independently movable child/u);
    expect(fs.existsSync(`${fixture.backupWorld}.swap-journal.json`)).toBe(false);
  });

  test("rejects a candidate that drops archived current-demo lineage before any transaction state is written", () => {
    const fixture = makeFixture();
    const manifest = JSON.parse(fs.readFileSync(fixture.candidateManifestPath));
    delete manifest.candidateBuild.lineageScope;
    writeJson(fixture.candidateManifestPath, manifest);
    fixture.candidateManifestSha = sha256File(fixture.candidateManifestPath);
    const adoptionPath = path.join(
      fixture.candidateWorld,
      "qa",
      "strict-clean-scene-adoption-report.json",
    );
    const adoption = JSON.parse(fs.readFileSync(adoptionPath));
    adoption.outputManifest.sha256 = fixture.candidateManifestSha;
    writeJson(adoptionPath, adoption);
    const qa = JSON.parse(fs.readFileSync(fixture.candidateBrowserQaReport));
    qa.manifest.sha256Before = fixture.candidateManifestSha;
    qa.manifest.sha256After = fixture.candidateManifestSha;
    writeJson(fixture.candidateBrowserQaReport, qa);

    expect(() => promote(fixture)).toThrow(
      /candidateBuild\.lineageScope must equal archived_current_demo_only/u,
    );
    expect(fs.existsSync(`${fixture.backupWorld}.swap-journal.json`)).toBe(false);
  });

  test("refuses a concurrent promotion while the canonical transaction lock is active", () => {
    const fixture = makeFixture();
    const lockPath = path.join(
      path.dirname(fixture.canonicalWorld),
      `.${path.basename(fixture.canonicalWorld)}.strict-clean-promotion.lock.json`,
    );
    writeJson(lockPath, {
      schemaVersion: 1,
      kind: "video2world.bedroom4_strict_clean_scene_promotion_lock",
      token: "active-fixture-lock",
      pid: process.pid,
      hostname: os.hostname(),
      startedAt: new Date().toISOString(),
      canonicalWorld: fixture.canonicalWorld,
      backupWorld: fixture.backupWorld,
      journalPath: `${fixture.backupWorld}.swap-journal.json`,
    });

    expect(() => promote(fixture)).toThrow(/promotion lock already held/u);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
    expect(fs.existsSync(fixture.backupWorld)).toBe(false);
  });

  test("rejects QA evidence after a bound runtime implementation changes", () => {
    const fixture = makeFixture();
    const runtimePath = path.join(fixture.root, "project", "web", "visual-physics-proxy.js");
    const original = fs.readFileSync(runtimePath);
    fs.unlinkSync(runtimePath);
    writeBytes(runtimePath, Buffer.concat([original, Buffer.from("\n// tampered fixture\n")]));

    expect(() => promote(fixture)).toThrow(/runtime: implementation size mismatch/u);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
  });

  test("rejects a screenshot with forged descriptor hashes but invalid PNG CRC", () => {
    const fixture = makeFixture();
    const qa = JSON.parse(fs.readFileSync(fixture.candidateBrowserQaReport));
    const screenshotPath = qa.desktop.screenshot.path;
    const bytes = fs.readFileSync(screenshotPath);
    const idat = bytes.indexOf(Buffer.from("IDAT", "ascii"));
    expect(idat).toBeGreaterThan(0);
    bytes[idat + 8] ^= 0x01;
    writeBytes(screenshotPath, bytes);
    qa.desktop.screenshot.sha256 = sha256File(screenshotPath);
    qa.desktop.screenshot.size = bytes.length;
    writeJson(fixture.candidateBrowserQaReport, qa);

    expect(() => promote(fixture)).toThrow(/PNG IDAT CRC mismatch/u);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
  });

  test("rejects forged hierarchy evidence even when the report claims passed", () => {
    const fixture = makeFixture();
    const qa = JSON.parse(fs.readFileSync(fixture.candidateBrowserQaReport));
    qa.desktop.hierarchy.parentMotion.sam3_pillow_front.relativeParentMatrixDelta = 0.25;
    writeJson(fixture.candidateBrowserQaReport, qa);

    expect(() => promote(fixture)).toThrow(/relative parent transform drifted/u);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
  });

  test("rejects forged bed shape drift beyond the serialized-matrix tolerance", () => {
    const fixture = makeFixture();
    const qa = JSON.parse(fs.readFileSync(fixture.candidateBrowserQaReport));
    qa.desktop.hierarchy.parentMotion.sam3_bed_01.splatMatrixWorld.gramDelta = 0.001;
    writeJson(fixture.candidateBrowserQaReport, qa);

    expect(() => promote(fixture)).toThrow(/parent motion changed sam3_bed_01\.splatMatrixWorld shape/u);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
    expect(fs.existsSync(fixture.backupWorld)).toBe(false);
  });

  test("rejects a candidate whose fixed stable alias asset closure is incomplete", () => {
    const fixture = makeFixture();
    const alias = JSON.parse(fs.readFileSync(
      path.join(fixture.candidateWorld, "manifest.web-demo-baseline-stable.json"),
    ));
    const stableUrl = collectPrefixUrls(alias, STABLE_PREFIX)[0];
    const relative = stableUrl.slice(`${STABLE_PREFIX}/`.length);
    fs.unlinkSync(path.join(fixture.candidateWorld, ...relative.split("/")));

    expect(() => promote(fixture)).toThrow(/stable alias URL is missing/u);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
  });

  test("detects an equal-size asset mutation during swap and restores the old canonical", () => {
    const fixture = makeFixture();
    let calls = 0;
    const renameDirectory = (source, target) => {
      calls += 1;
      if (calls === 2) {
        const assetPath = path.join(source, "chunks", "static-visual.chunk000");
        const bytes = fs.readFileSync(assetPath);
        bytes[0] ^= 0xff;
        fs.writeFileSync(assetPath, bytes);
      }
      fs.renameSync(source, target);
    };

    expect(() => promote(fixture, { renameDirectory })).toThrow(AggregateError);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
    expect(fs.existsSync(fixture.candidateWorld)).toBe(true);
    expect(fs.existsSync(fixture.backupWorld)).toBe(false);
  });

  test("CLI recover rolls back a crash after both renames using the durable journal", () => {
    const fixture = makeFixture();
    const result = promote(fixture);
    const journal = JSON.parse(fs.readFileSync(result.journalPath));
    journal.status = "in_progress";
    journal.phase = "candidate_moved_to_canonical";
    delete journal.receiptSha256;
    writeJson(result.journalPath, journal);

    const output = JSON.parse(execFileSync(
      process.execPath,
      [promotionScript, "--recover-journal", result.journalPath],
      { encoding: "utf8" },
    ));

    expect(output).toMatchObject({
      status: "recovered_rolled_back",
      phase: "recovered_rolled_back",
    });
    expect(fs.existsSync(fixture.canonicalWorld)).toBe(true);
    expect(fs.existsSync(fixture.candidateWorld)).toBe(true);
    expect(fs.existsSync(fixture.backupWorld)).toBe(false);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
    expect(sha256File(fixture.candidateManifestPath)).toBe(fixture.candidateManifestSha);
    expect(fs.existsSync(path.join(
      fixture.candidateWorld,
      "qa",
      "strict-clean-scene-swap-receipt.json",
    ))).toBe(false);
    expect(fs.readdirSync(path.join(fixture.candidateWorld, "qa"))
      .some((name) => name.startsWith("strict-clean-scene-swap-receipt.json.aborted-")))
      .toBe(true);
  }, 15_000);

  test("CLI recover rolls back a crash after only canonical moved to backup", () => {
    const fixture = makeFixture();
    const result = promote(fixture);
    fs.renameSync(fixture.canonicalWorld, fixture.candidateWorld);
    const journal = JSON.parse(fs.readFileSync(result.journalPath));
    journal.status = "in_progress";
    journal.phase = "canonical_moved_to_backup";
    delete journal.receiptSha256;
    writeJson(result.journalPath, journal);

    const output = JSON.parse(execFileSync(
      process.execPath,
      [promotionScript, "--recover-journal", result.journalPath],
      { encoding: "utf8" },
    ));

    expect(output.status).toBe("recovered_rolled_back");
    expect(fs.existsSync(fixture.canonicalWorld)).toBe(true);
    expect(fs.existsSync(fixture.candidateWorld)).toBe(true);
    expect(fs.existsSync(fixture.backupWorld)).toBe(false);
    expect(sha256File(fixture.publicManifest)).toBe(fixture.canonicalManifestSha);
    expect(sha256File(fixture.candidateManifestPath)).toBe(fixture.candidateManifestSha);
  }, 15_000);
  },
);
