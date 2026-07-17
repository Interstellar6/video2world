import { describe, expect, it } from "vitest";
import { fileURLToPath } from "node:url";

import {
  APPROVED_NESTED_SUPPORT,
  EXPECTED_TOPOLOGY,
  UNIFIED_IDS,
  buildRuntimeUrl,
  cameraTargetBoundsEvidence,
  classifyInterpenetrations,
  deriveManifestExpectations,
  parseArgs,
  usage,
} from "../../scripts/qa_bedroom4_clean_unified_scene_browser.mjs";

const LEGACY_IDS = [
  "sam3_nightstand_01",
  "sam3_nightstand_02",
  "sam3_plant_01",
  "sam3_plant_02",
];

function sha(index) {
  return index.toString(16).padStart(64, "0");
}

function unifiedObject(id, index) {
  const bed = id === "sam3_bed_01";
  return {
    id,
    independentlyMovable: true,
    semanticGranularity: bed ? "independent_root_asset" : "independent_child_asset",
    parentObjectId: bed ? null : "sam3_bed_01",
    movesWithParent: !bed,
    visual: null,
    renderAsset: null,
    colliderProxy: null,
    interaction: {
      kind: "spin",
      drag: "horizontal_yaw",
      degrees: 360,
      durationMs: 10,
    },
    collision: {
      mode: "unified-glb",
      topology: EXPECTED_TOPOLOGY[id],
      characterCollision: true,
      gate: { status: "passed" },
      asset: {
        url: `./worlds/bedroom4/objects/${id}.glb`,
        sha256: sha(index + 20),
        vertices: 100 + index,
        faces: 20 + index,
        finite: true,
        nondegenerate: true,
        windingConsistent: true,
        watertight: bed,
      },
    },
    limitations: index % 2 ? [`minor fixture limitation ${index}`] : [],
  };
}

function legacyObject(id, index) {
  return {
    id,
    independentlyMovable: true,
    visual: {
      format: "graphdeco-gaussian-ply",
      vertexCount: 10 * (index + 1),
    },
    interaction: { kind: "spin", degrees: 360, durationMs: 10 },
    collision: {
      mode: "kinematic",
      characterCollision: true,
      gate: { status: "passed" },
      asset: { faces: 11 + index },
    },
  };
}

function fixtureManifest() {
  const interactiveObjects = [
    ...LEGACY_IDS.map(legacyObject),
    ...UNIFIED_IDS.map(unifiedObject),
  ];
  const bed = interactiveObjects.find((item) => item.id === "sam3_bed_01");
  bed.childObjectIds = [...APPROVED_NESTED_SUPPORT.childObjectIds];
  bed.collision.gate.supportAdjustmentReceiptSha256 = APPROVED_NESTED_SUPPORT.receiptSha256;
  return {
    initialState: { cameraFocusObjectId: "sam3_bed_01" },
    assets: {
      visual: { vertexCount: 111 },
      cleanTsdf: { faceCount: 222 },
    },
    collisionWorld: { sceneAssetKey: "cleanTsdf" },
    interactiveObjects,
    sceneKnowledge: {
      objects: UNIFIED_IDS.map((id) => ({
        id,
        bbox: { min: [0, 0, 0], max: [1, 1, 1] },
        description: { short: { zh: `${id} fixture` } },
      })),
    },
  };
}

function bounds(min, max) {
  return {
    min: { x: min[0], y: min[1], z: min[2] },
    max: { x: max[0], y: max[1], z: max[2] },
  };
}

describe("clean unified scene browser QA manifest expectations", () => {
  it("accepts an already-correct focus target without requiring camera movement", () => {
    const inspection = {
      collisionBounds: {
        min: { x: -10, y: -5, z: -20 },
        max: { x: 10, y: 5, z: 20 },
        center: { x: 0, y: 0, z: 0 },
      },
    };
    expect(cameraTargetBoundsEvidence({ cameraTarget: { x: 0, y: 0, z: 0 } }, inspection))
      .toMatchObject({ passed: true, normalizedCenterDistance: 0, targetInsideBounds: true });
    expect(cameraTargetBoundsEvidence({ cameraTarget: { x: 9, y: 0, z: 0 } }, inspection).passed)
      .toBe(false);
    expect(cameraTargetBoundsEvidence({ cameraTarget: { x: 100, y: 0, z: 0 } }, inspection).passed)
      .toBe(false);
    expect(cameraTargetBoundsEvidence({ cameraTarget: null }, inspection).passed).toBe(false);
  });

  it("derives every count, face count, and vertex count from the manifest", () => {
    const manifest = fixtureManifest();
    const expected = deriveManifestExpectations(manifest);

    expect(expected).toMatchObject({
      cameraFocusObjectId: "sam3_bed_01",
      objectCount: 8,
      objectColliderCount: 8,
      staticVisualCount: 111,
      staticColliderFaces: 222,
      gaussianVertices: 100,
      rgbPointVertices: 0,
      meshVertices: 406,
      objectColliderFaces: 136,
      totalVisualPrimitives: 617,
      nestedSupport: APPROVED_NESTED_SUPPORT,
    });
    expect(Object.keys(expected.unified)).toEqual(UNIFIED_IDS);
    expect(expected.unified.sam3_pillow_front).toMatchObject({
      topology: "surface_bvh",
      vertices: 100,
      faces: 20,
      parentObjectId: "sam3_bed_01",
      movesWithParent: true,
    });
    expect(expected.unified.sam3_bed_01).toMatchObject({
      topology: "closed_volume",
      vertices: 103,
      faces: 23,
      parentObjectId: null,
      movesWithParent: false,
    });
    expect(expected.queryCases.map((item) => item.label)).toEqual([
      "pillow_location",
      "pillow_appearance",
      "bed_location",
      "bed_appearance",
    ]);

    manifest.interactiveObjects[0].visual.format = "rgb-point-cloud-ply";
    const withRgbPoints = deriveManifestExpectations(manifest);
    expect(withRgbPoints.gaussianVertices).toBe(90);
    expect(withRgbPoints.rgbPointVertices).toBe(10);
    expect(withRgbPoints.totalVisualPrimitives).toBe(617);

    manifest.assets.visual.vertexCount = 7;
    manifest.assets.cleanTsdf.faceCount = 9;
    manifest.interactiveObjects.find((item) => item.id === "sam3_bed_01")
      .collision.asset.vertices = 1000;
    const changed = deriveManifestExpectations(manifest);
    expect(changed.staticVisualCount).toBe(7);
    expect(changed.staticColliderFaces).toBe(9);
    expect(changed.meshVertices).toBe(1303);
    expect(changed.totalVisualPrimitives).toBe(1410);
  });

  it("fails closed on missing objects, wrong topology, or a unified proxy", () => {
    const missing = fixtureManifest();
    missing.interactiveObjects.pop();
    expect(() => deriveManifestExpectations(missing)).toThrow(/exactly 8/u);

    const topology = fixtureManifest();
    topology.interactiveObjects.find((item) => item.id === "sam3_bed_01")
      .collision.topology = "surface_bvh";
    expect(() => deriveManifestExpectations(topology)).toThrow(/closed_volume/u);

    const proxy = fixtureManifest();
    proxy.interactiveObjects.find((item) => item.id === "sam3_pillow_left")
      .colliderProxy = { type: "box" };
    expect(() => deriveManifestExpectations(proxy)).toThrow(/colliderProxy must be null/u);

    const duplicateUrl = fixtureManifest();
    duplicateUrl.interactiveObjects.find((item) => item.id === "sam3_pillow_left")
      .collision.asset.url = "./worlds/bedroom4/objects/../objects/sam3_pillow_front.glb";
    expect(() => deriveManifestExpectations(duplicateUrl)).toThrow(/GLB URL is duplicated/u);

    const wrongParent = fixtureManifest();
    wrongParent.interactiveObjects.find((item) => item.id === "sam3_pillow_left")
      .parentObjectId = null;
    expect(() => deriveManifestExpectations(wrongParent)).toThrow(/parentObjectId is invalid/u);
  });

  it("requires the manifest to frame the bed as the initial camera overview", () => {
    const missing = fixtureManifest();
    delete missing.initialState.cameraFocusObjectId;
    expect(() => deriveManifestExpectations(missing)).toThrow(
      /initialState\.cameraFocusObjectId must be sam3_bed_01/u,
    );

    const wrongObject = fixtureManifest();
    wrongObject.initialState.cameraFocusObjectId = "sam3_pillow_front";
    expect(() => deriveManifestExpectations(wrongObject)).toThrow(
      /initialState\.cameraFocusObjectId must be sam3_bed_01/u,
    );
  });

  it("parses the side-effect-free CLI and rejects ambiguous options", () => {
    expect(parseArgs([
      "--url", "http://127.0.0.1:4177/",
      "--manifest", "/tmp/manifest.json",
      "--report", "/tmp/report.json",
      "--screenshot-dir", "/tmp/screens",
      "--min-fps", "21.5",
    ])).toMatchObject({
      url: "http://127.0.0.1:4177/",
      manifest: "/tmp/manifest.json",
      report: "/tmp/report.json",
      "screenshot-dir": "/tmp/screens",
      minFps: 21.5,
    });
    expect(parseArgs(["--help"])).toEqual({ help: true });
    expect(() => parseArgs(["--manifest"])).toThrow(/Missing value/u);
    expect(() => parseArgs(["--unknown", "x"])).toThrow(/Unsupported option/u);
    expect(() => parseArgs(["--report", "a", "--report", "b"])).toThrow(/Duplicate option/u);
    expect(() => parseArgs(["--min-fps", "0"])).toThrow(/positive finite/u);
    expect(usage()).toContain("never edits or promotes the manifest");
  });

  it("keeps manifest and GLB URLs relative to a deployed subpath", () => {
    const manifestPath = fileURLToPath(new URL(
      "../../web/public/worlds/bedroom4/manifest.fixture.json",
      import.meta.url,
    ));
    const runtime = new URL(buildRuntimeUrl(
      "https://relumeow.top/video2world/web-demo/",
      manifestPath,
    ));
    const manifestUrl = runtime.searchParams.get("manifest");

    expect(manifestUrl).toBe("./worlds/bedroom4/manifest.fixture.json");
    expect(new URL(manifestUrl, runtime).pathname).toBe(
      "/video2world/web-demo/worlds/bedroom4/manifest.fixture.json",
    );
    expect(new URL("./worlds/bedroom4/objects/pillow.glb", runtime).pathname).toBe(
      "/video2world/web-demo/worlds/bedroom4/objects/pillow.glb",
    );
  });

  it("records support and minor contacts but blocks only obvious object interpenetration", () => {
    const report = classifyInterpenetrations({
      status: "failed",
      intersections: [
        {
          colliderIds: ["sam3_bed_01", "sam3_pillow_front"],
          leftBounds: bounds([0, 0, 0], [2, 2, 2]),
          rightBounds: bounds([0.2, 1.98, 0.2], [1.8, 2.98, 1.8]),
        },
        {
          colliderIds: ["sam3_pillow_left", "sam3_pillow_right"],
          leftBounds: bounds([0, 0, 0], [1, 1, 1]),
          rightBounds: bounds([0.98, 0.98, 0.98], [1.98, 1.98, 1.98]),
        },
        {
          colliderIds: ["sam3_nightstand_01", "sam3_pillow_front"],
          leftBounds: bounds([0, 0, 0], [2, 2, 2]),
          rightBounds: bounds([0.25, 0.25, 0.25], [1.75, 1.75, 1.75]),
        },
        {
          colliderIds: ["sam3_bed_01", "scene"],
          leftBounds: bounds([-10, -10, -10], [10, 10, 10]),
          rightBounds: bounds([0, 0, 0], [1, 1, 1]),
        },
        {
          colliderIds: ["sam3_bed_01", "sam3_pillow_front"],
          leftBounds: bounds([0, 0, 0], [2, 2, 2]),
          rightBounds: bounds([0.2, 0.2, 0.2], [1.8, 1.8, 1.8]),
        },
      ],
    });

    expect(report.automatedStatus).toBe("failed_obvious_interpenetration");
    expect(report.classifications.map((item) => [item.classification, item.blocking])).toEqual([
      ["bed_pillow_support_contact_record_only", false],
      ["minor_or_conservative_unified_object_contact", false],
      ["obvious_unified_object_interpenetration", true],
      ["scene_support_or_surface_contact_record_only", false],
      ["obvious_unified_object_interpenetration", true],
    ]);
    expect(report.policy.currentUserTolerance).toContain("minor_contact");
    expect(report.policy.humanScreenshotReviewStillRequired).toBe(true);

    const exactReceiptBoundSupport = classifyInterpenetrations({
      status: "failed",
      intersections: [{
        colliderIds: ["sam3_bed_01", "sam3_pillow_front"],
        leftBounds: bounds([0, 0, 0], [2, 2, 2]),
        rightBounds: bounds([0.2, 0.2, 0.2], [1.8, 1.8, 1.8]),
      }],
    }, APPROVED_NESTED_SUPPORT);
    expect(exactReceiptBoundSupport.automatedStatus).toBe("passed_with_recorded_contacts");
    expect(exactReceiptBoundSupport.classifications[0]).toMatchObject({
      classification: "validated_bed_pillow_support_contact_record_only",
      blocking: false,
      nestedSupportReceiptSha256: APPROVED_NESTED_SUPPORT.receiptSha256,
    });
  });
});
