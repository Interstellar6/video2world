import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, test } from "vitest";

import {
  CLEAN_PLATE_UNIFIED_OBJECT_IDS,
  LEGACY_CARVE_OBJECT_IDS,
  materializeStrictCleanScene,
} from "../../scripts/materialize_bedroom4_strict_clean_scene.mjs";
import { sha256, sha256File } from "../../scripts/lib/strict-clean-scene-assets.mjs";
import { validateWebManifest } from "../../web/web-manifest.js";

const roots = [];
const SOURCE_URL_PREFIX = "./worlds/bedroom4";
const OUTPUT_URL_PREFIX = "./worlds/bedroom4-strict-clean-candidate";
const TARGET_FRAME = "pgsr_native_shared_frame_20260714";
const PILLOW_IDS = CLEAN_PLATE_UNIFIED_OBJECT_IDS.slice(0, 3);
const BED_MATRIX_ROW_MAJOR = Object.freeze([
  Object.freeze([6.0940405667591, 0.06831981888678311, 2.648131223143048, 0.639241276163562]),
  Object.freeze([-0.20072433617157628, -12.444750282849224, 0.9540323820267776, 5.6921053762733935]),
  Object.freeze([1.9936905918532148, -1.4617654309371404, -7.998393355783393, 10.762354974571135]),
  Object.freeze([0, 0, 0, 1]),
]);

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

function writeBytes(filePath, bytes) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, bytes);
  return bytes;
}

function writeJson(filePath, value) {
  writeBytes(filePath, Buffer.from(`${JSON.stringify(value, null, 2)}\n`));
}

function fileAsset(url, filePath, extra = {}) {
  return {
    url,
    size: fs.statSync(filePath).size,
    sha256: sha256File(filePath),
    ...extra,
  };
}

function chunkedAsset(urls, paths, extra = {}) {
  const bytes = paths.map((filePath) => fs.readFileSync(filePath));
  const whole = Buffer.concat(bytes);
  return {
    size: whole.length,
    sha256: sha256(whole),
    parts: paths.map((filePath, index) => ({
      url: urls[index],
      size: fs.statSync(filePath).size,
      sha256: sha256File(filePath),
    })),
    ...extra,
  };
}

function graphdecoPly(positions) {
  const properties = [
    "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
  ];
  const header = [
    "ply",
    "format binary_little_endian 1.0",
    `element vertex ${positions.length}`,
    ...properties.map((name) => `property float ${name}`),
    "end_header",
    "",
  ].join("\n");
  const payload = Buffer.alloc(positions.length * properties.length * 4);
  positions.forEach((position, vertex) => {
    const values = [
      ...position,
      0, 0, 0,
      0.5, 0.5, 0.5,
      1,
      0, 0, 0,
      1, 0, 0, 0,
    ];
    values.forEach((value, property) => {
      payload.writeFloatLE(value, (vertex * properties.length + property) * 4);
    });
  });
  return Buffer.concat([Buffer.from(header, "ascii"), payload]);
}

function tsdfPly(triangleCenters) {
  const vertices = triangleCenters.flatMap(([x, y, z]) => [
    [x - 0.1, y, z],
    [x + 0.1, y, z],
    [x, y + 0.1, z],
  ]);
  const header = [
    "ply",
    "format binary_little_endian 1.0",
    `element vertex ${vertices.length}`,
    "property double x",
    "property double y",
    "property double z",
    "property uchar red",
    "property uchar green",
    "property uchar blue",
    `element face ${triangleCenters.length}`,
    "property list uchar uint vertex_indices",
    "end_header",
    "",
  ].join("\n");
  const vertexBytes = Buffer.alloc(vertices.length * 27);
  vertices.forEach((position, index) => {
    const offset = index * 27;
    vertexBytes.writeDoubleLE(position[0], offset);
    vertexBytes.writeDoubleLE(position[1], offset + 8);
    vertexBytes.writeDoubleLE(position[2], offset + 16);
    vertexBytes[offset + 24] = 200;
    vertexBytes[offset + 25] = 200;
    vertexBytes[offset + 26] = 200;
  });
  const faceBytes = Buffer.alloc(triangleCenters.length * 13);
  triangleCenters.forEach((_center, index) => {
    const offset = index * 13;
    faceBytes[offset] = 3;
    faceBytes.writeUInt32LE(index * 3, offset + 1);
    faceBytes.writeUInt32LE(index * 3 + 1, offset + 5);
    faceBytes.writeUInt32LE(index * 3 + 2, offset + 9);
  });
  return Buffer.concat([Buffer.from(header, "ascii"), vertexBytes, faceBytes]);
}

function localUrlPath(url, prefix, worldDir) {
  expect(url.startsWith(`${prefix}/`)).toBe(true);
  return path.join(worldDir, ...url.slice(prefix.length + 1).split("/"));
}

function allWorldUrls(value, prefix, output = []) {
  if (typeof value === "string") {
    if (value.startsWith(`${prefix}/`)) output.push(value);
  } else if (Array.isArray(value)) {
    value.forEach((item) => allWorldUrls(item, prefix, output));
  } else if (value && typeof value === "object") {
    Object.values(value).forEach((item) => allWorldUrls(item, prefix, output));
  }
  return output;
}

function makeLegacyObject({ root, sourceWorld, id, center, index }) {
  const visualA = writeBytes(
    path.join(sourceWorld, "chunks", `${id}.visual.chunk000`),
    Buffer.from(`visual-${id}-a`),
  );
  const visualB = writeBytes(
    path.join(sourceWorld, "chunks", `${id}.visual.chunk001`),
    Buffer.from(`visual-${id}-b`),
  );
  const colliderPath = path.join(sourceWorld, "objects", `${id}.collider.glb`);
  const renderPath = path.join(sourceWorld, "objects", `${id}.render.glb`);
  const anchorPath = path.join(sourceWorld, "evidence", `${id}.source-anchor.ply`);
  writeBytes(colliderPath, Buffer.from(`glTF-collider-${id}`));
  writeBytes(renderPath, Buffer.from(`glTF-render-${id}`));
  writeBytes(anchorPath, Buffer.from(`ply-anchor-${id}`));
  const visualPaths = [
    path.join(sourceWorld, "chunks", `${id}.visual.chunk000`),
    path.join(sourceWorld, "chunks", `${id}.visual.chunk001`),
  ];
  const visualUrls = [
    `${SOURCE_URL_PREFIX}/chunks/${id}.visual.chunk000`,
    `${SOURCE_URL_PREFIX}/chunks/${id}.visual.chunk001`,
  ];
  expect(Buffer.concat([visualA, visualB]).length).toBeGreaterThan(0);
  return {
    id,
    placement: { pivot: [center, 0, 0], scale: [1, 1, 1], rotationEulerDeg: [0, 0, 0] },
    visual: chunkedAsset(visualUrls, visualPaths, {
      id: `${id}_visual`,
      fileName: `${id}.ply`,
      fileType: "ply",
      format: "graphdeco-gaussian-ply",
    }),
    colliderProxy: { type: "selection-box", center: [0, 0, 0], dimensions: [1, 1, 1] },
    collision: {
      mode: "kinematic",
      characterCollision: true,
      gate: { status: "passed" },
      asset: fileAsset(`${SOURCE_URL_PREFIX}/objects/${id}.collider.glb`, colliderPath, {
        fileName: `${id}.collider.glb`,
        fileType: "glb",
        format: "gltf-binary",
      }),
      renderAsset: fileAsset(`${SOURCE_URL_PREFIX}/objects/${id}.render.glb`, renderPath, {
        fileName: `${id}.render.glb`,
        fileType: "glb",
        format: "gltf-binary",
      }),
    },
    interaction: { kind: "spin", degrees: 360, durationMs: 1000 },
    sourceAnchor: {
      path: `${SOURCE_URL_PREFIX}/evidence/${id}.source-anchor.ply`,
      size: fs.statSync(anchorPath).size,
      sha256: sha256File(anchorPath),
      robustBounds: {
        min: [center - 0.5, -0.5, -0.5],
        max: [center + 0.5, 0.5, 0.5],
      },
    },
    semanticGranularity: "independent_root_asset",
    parentObjectId: null,
    movesWithParent: false,
    independentlyMovable: true,
    fixtureIndex: index,
  };
}

function makeUnifiedObject({ projectRoot, id, center }) {
  const isPillow = id !== "sam3_bed_01";
  const fileName = `${id}.glb`;
  const source = path.join(projectRoot, "fixture-assets", fileName);
  writeBytes(source, Buffer.from(`glTF-unified-${id}`));
  const placement = isPillow
    ? {
        assetCoordinatesBaked: true,
        pivot: [center, 0, 0],
        scale: [1, 1, 1],
        rotationEulerDeg: [0, 0, 0],
        generatedCenter: [0, 0, 0],
      }
    : {
        assetCoordinatesBaked: false,
        pivot: BED_MATRIX_ROW_MAJOR.slice(0, 3).map((row) => row[3]),
        scale: [1, 1, 1],
        rotationEulerDeg: [0, 0, 0],
        generatedCenter: [0, 0, 0],
        matrixRowMajor: BED_MATRIX_ROW_MAJOR.map((row) => [...row]),
      };
  return {
    id,
    placement,
    collision: {
      mode: "unified-glb",
      topology: isPillow ? "surface_bvh" : "closed_volume",
      characterCollision: true,
      gate: { status: "passed", surfaceCollision: "passed", candidateBrowserQa: "pending" },
      asset: fileAsset(`${SOURCE_URL_PREFIX}/objects/${fileName}`, source, {
        id: `${id}_unified`,
        fileName,
        fileType: "glb",
        format: "gltf-binary",
        sourcePath: `repo://video2world/fixture-assets/${fileName}`,
        faces: 12,
        finite: true,
        nondegenerate: true,
        windingConsistent: true,
        watertight: !isPillow,
      }),
    },
    interaction: { kind: "spin", degrees: 360, durationMs: 1000, drag: "horizontal_yaw" },
    semanticGranularity: isPillow ? "independent_child_asset" : "independent_root_asset",
    parentObjectId: isPillow ? "sam3_bed_01" : null,
    movesWithParent: isPillow,
    independentlyMovable: true,
    childObjectIds: isPillow ? [] : [...PILLOW_IDS],
    limitations: ["Candidate asset transport and browser QA are pending."],
  };
}

function baseManifest(assets, interactiveObjects = []) {
  return {
    schemaVersion: 1,
    contract: "video2world-web-manifest-1.0.0",
    version: "fixture",
    coordinateSystem: { worldUp: [0, -1, 0] },
    assets,
    interactiveObjects,
    sceneKnowledge: { objects: [] },
    presentation: { referenceBoundsAssetKey: "collider" },
  };
}

function makeFixture({ emptyOutput = false } = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "video2world-strict-clean-"));
  roots.push(root);
  const projectRoot = path.join(root, "project");
  const sourceWorld = path.join(projectRoot, "web", "public", "worlds", "bedroom4");
  const outputWorld = path.join(
    projectRoot,
    "web",
    "public",
    "worlds",
    "bedroom4-strict-clean-candidate",
  );
  fs.mkdirSync(sourceWorld, { recursive: true });
  if (emptyOutput) fs.mkdirSync(outputWorld, { recursive: true });

  const baselineVisualPath = path.join(sourceWorld, "baseline", "visual.ply");
  const baselineColliderPath = path.join(sourceWorld, "baseline", "collider.ply");
  writeBytes(baselineVisualPath, Buffer.from("baseline-visual"));
  writeBytes(baselineColliderPath, Buffer.from("baseline-collider"));
  writeBytes(path.join(sourceWorld, "qa", "browser-qa.json"), Buffer.from("{}\n"));
  writeBytes(path.join(sourceWorld, "baseline-marker.txt"), Buffer.from("baseline-world-marker"));
  const baselineAssets = {
    visual: fileAsset(`${SOURCE_URL_PREFIX}/baseline/visual.ply`, baselineVisualPath, {
      fileName: "visual.ply",
      fileType: "ply",
      format: "graphdeco-gaussian-ply",
    }),
    collider: fileAsset(`${SOURCE_URL_PREFIX}/baseline/collider.ply`, baselineColliderPath, {
      fileName: "collider.ply",
      fileType: "ply",
      format: "open3d-binary-little-endian-triangle-mesh-ply",
    }),
  };
  const publicManifestPath = path.join(sourceWorld, "manifest.json");
  writeJson(publicManifestPath, baseManifest(baselineAssets));
  const publicSha = sha256File(publicManifestPath);

  const centers = [0, 10, 20, 30];
  const legacy = LEGACY_CARVE_OBJECT_IDS.map((id, index) => makeLegacyObject({
    root,
    sourceWorld,
    id,
    center: centers[index],
    index,
  }));
  const unified = CLEAN_PLATE_UNIFIED_OBJECT_IDS.map((id, index) => makeUnifiedObject({
    projectRoot,
    id,
    center: 100 + index * 10,
  }));
  const candidate = {
    ...baseManifest(baselineAssets, [...legacy, ...unified]),
    version: "fixture-four-unified-candidate",
    sourceWorld: {
      worldId: "bedroom_4",
      runId: "fixture-run",
      adoptionMode: "fixture-baseline",
    },
    candidateBuild: {
      cleanScene: {
        status: "clean_scene_pending",
        methodRequired: "front_to_back_cumulative_peel_then_clean_plate_reconstruction",
      },
      inheritedStaticScene: { baseManifestSha256: publicSha, modifiedByThisCandidate: false },
      unifiedPbrObjects: {
        objectIds: [...CLEAN_PLATE_UNIFIED_OBJECT_IDS],
        status: "geometry_and_joint_qa_passed",
        hierarchy: {
          parent: "sam3_bed_01",
          children: CLEAN_PLATE_UNIFIED_OBJECT_IDS.slice(0, 3),
          parentMotionCarriesChildren: true,
          childrenRemainIndependentlyMovable: true,
        },
      },
      promotionAllowed: false,
    },
  };
  const candidateManifest = path.join(projectRoot, "candidate.json");
  writeJson(candidateManifest, candidate);

  const evidenceDir = path.join(projectRoot, "fresh-run");
  const pgsrPly = path.join(evidenceDir, "point_cloud.ply");
  const tsdfPly = path.join(evidenceDir, "tsdf_fusion_post.ply");
  writeBytes(pgsrPly, graphdecoPly([
    [0, 0, 0],
    [10, 0, 0],
    [20, 0, 0],
    [30, 0, 0],
    [40, 0, 0],
    [41, 1, 1],
  ]));
  writeBytes(tsdfPly, tsdfPlyBytes([
    [0, 0, 0],
    [10, 0, 0],
    [20, 0, 0],
    [30, 0, 0],
    [40, 0, 0],
  ]));

  const receiptInputs = {};
  for (const key of [
    "strict_sequence_report",
    "reconstruction_input_manifest",
    "reference_cameras",
    "fresh_cameras",
    "pgsr_stage_receipt",
    "tsdf_stage_receipt",
  ]) {
    const filePath = path.join(evidenceDir, `${key}.json`);
    writeJson(filePath, { fixture: key });
    receiptInputs[key] = { path: filePath, sha256: sha256File(filePath) };
  }
  receiptInputs.pgsr_ply = { path: pgsrPly, sha256: sha256File(pgsrPly) };
  receiptInputs.tsdf_ply = { path: tsdfPly, sha256: sha256File(tsdfPly) };
  const canonicalCameraSha = sha256(Buffer.from("canonical-camera-subset"));
  const alignmentReceipt = path.join(evidenceDir, "clean_scene_alignment_receipt.json");
  writeJson(alignmentReceipt, {
    schema_version: 1,
    kind: "video2world.clean_scene_alignment_receipt",
    status: "passed_current_demo_only",
    acceptance_scope: "current_demo_only",
    promotion_approved: false,
    source_coordinate_frame: TARGET_FRAME,
    target_coordinate_frame: TARGET_FRAME,
    method: "identity_same_camera_calibration",
    transform: {
      scale: 1,
      rotationMatrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      translation: [0, 0, 0],
    },
    canonical_camera_subset_sha256: canonicalCameraSha,
    camera_comparison: {
      reference_camera_count: 3,
      fresh_camera_count: 2,
      fresh_img_names: ["000048", "000052"],
      absolute_tolerance: 1e-12,
      maximum_absolute_delta: 0,
    },
    inputs: receiptInputs,
    gates: {
      camera_calibration_hash_bound: true,
      reference_subset_exact: true,
      pgsr_and_tsdf_share_frame: true,
      object_placements_share_target_frame: true,
      transform_finite: true,
      identity_or_verified_similarity: true,
      source_stage_receipts_unchanged: true,
    },
  });
  return {
    root,
    projectRoot,
    sourceWorld,
    outputWorld,
    publicManifestPath,
    publicSha,
    candidateManifest,
    pgsrPly,
    tsdfPly,
    alignmentReceipt,
    alignmentReceiptSha: sha256File(alignmentReceipt),
    canonicalCameraSha,
  };
}

// Avoid shadowing the fixture path variable while retaining the domain-specific helper name above.
const tsdfPlyBytes = tsdfPly;

function materialize(fixture, overrides = {}) {
  return materializeStrictCleanScene({
    projectRoot: fixture.projectRoot,
    candidateManifest: fixture.candidateManifest,
    pgsrPly: fixture.pgsrPly,
    tsdfPly: fixture.tsdfPly,
    alignmentReceipt: fixture.alignmentReceipt,
    alignmentReceiptSha256: fixture.alignmentReceiptSha,
    expectedCameraCanonicalSha256: fixture.canonicalCameraSha,
    expectedTargetCoordinateFrame: TARGET_FRAME,
    sourceWorldDir: fixture.sourceWorld,
    publicManifest: fixture.publicManifestPath,
    outputWorld: fixture.outputWorld,
    sourceUrlPrefix: SOURCE_URL_PREFIX,
    urlPrefix: OUTPUT_URL_PREFIX,
    storageMode: "hardlink",
    chunkSize: 96,
    carveMargin: 0,
    ...overrides,
  });
}

function mutateCandidate(fixture, mutate) {
  const manifest = JSON.parse(fs.readFileSync(fixture.candidateManifest, "utf8"));
  const objects = new Map(manifest.interactiveObjects.map((object) => [object.id, object]));
  mutate(objects, manifest);
  writeJson(fixture.candidateManifest, manifest);
}

describe("strict layered clean-scene adoption materializer", () => {
  test("builds a complete isolated world and carves only the four legacy objects", () => {
    const fixture = makeFixture({ emptyOutput: true });
    const sourceManifestBytes = fs.readFileSync(fixture.publicManifestPath);
    const sourceManifestInode = fs.statSync(fixture.publicManifestPath).ino;

    const result = materialize(fixture);

    expect(result.report.status).toBe("candidate_materialized_browser_qa_pending");
    expect(result.report.promotionApproved).toBe(false);
    expect(result.report.browserQa).toBe("pending");
    expect(fs.readFileSync(fixture.publicManifestPath)).toEqual(sourceManifestBytes);
    expect(sha256File(fixture.publicManifestPath)).toBe(fixture.publicSha);
    expect(fs.statSync(fixture.publicManifestPath).ino).toBe(sourceManifestInode);
    expect(fs.statSync(path.join(fixture.outputWorld, "manifest.json")).ino).not.toBe(sourceManifestInode);

    const manifest = JSON.parse(fs.readFileSync(path.join(fixture.outputWorld, "manifest.json")));
    expect(() => validateWebManifest(manifest)).not.toThrow();
    expect(manifest.candidateBuild.cleanScene.status)
      .toBe("strict_layered_clean_scene_materialized_current_demo_only");
    expect(manifest.candidateBuild.cleanScene.staticLegacyCarveObjectIds)
      .toEqual(LEGACY_CARVE_OBJECT_IDS);
    expect(manifest.candidateBuild.cleanScene.cleanPlateObjectIdsNotCarvedAgain)
      .toEqual(CLEAN_PLATE_UNIFIED_OBJECT_IDS);
    expect(manifest.candidateBuild.cleanScene.alignmentReceiptSha256)
      .toBe(fixture.alignmentReceiptSha);
    expect(manifest.alignment.canonicalCameraSubsetSha256).toBe(fixture.canonicalCameraSha);

    for (const descriptor of [manifest.assets.visual, manifest.assets.colliderStaticCarved]) {
      expect(descriptor.carvedObjectIds).toEqual(LEGACY_CARVE_OBJECT_IDS);
      expect(descriptor.protectedCleanPlateObjectIds).toEqual(CLEAN_PLATE_UNIFIED_OBJECT_IDS);
      expect(Object.keys(descriptor.removedByObject)).toEqual(LEGACY_CARVE_OBJECT_IDS);
      expect(Object.values(descriptor.removedByObject)).toEqual([1, 1, 1, 1]);
      for (const objectId of CLEAN_PLATE_UNIFIED_OBJECT_IDS) {
        expect(descriptor.removedByObject).not.toHaveProperty(objectId);
      }
    }
    expect(manifest.assets.visual.inputVertexCount).toBe(6);
    expect(manifest.assets.visual.vertexCount).toBe(2);
    expect(manifest.assets.colliderStaticCarved.originalFaceCount).toBe(5);
    expect(manifest.assets.colliderStaticCarved.faceCount).toBe(1);
    expect(manifest.assets.visual.parts.length).toBeGreaterThan(1);
    expect(manifest.assets.collider.parts.length).toBeGreaterThan(1);

    const objects = new Map(manifest.interactiveObjects.map((object) => [object.id, object]));
    for (const objectId of CLEAN_PLATE_UNIFIED_OBJECT_IDS) {
      const object = objects.get(objectId);
      expect(object.collision.mode).toBe("unified-glb");
      expect(object.visual).toBeUndefined();
      expect(object.colliderProxy).toBeUndefined();
      expect(object.collision.renderAsset).toBeUndefined();
      const target = localUrlPath(object.collision.asset.url, OUTPUT_URL_PREFIX, fixture.outputWorld);
      expect(sha256File(target)).toBe(object.collision.asset.sha256);
    }
    for (const objectId of PILLOW_IDS) {
      const pillow = objects.get(objectId);
      expect(pillow.placement).toMatchObject({
        assetCoordinatesBaked: true,
        scale: [1, 1, 1],
        rotationEulerDeg: [0, 0, 0],
      });
      expect(pillow.placement).not.toHaveProperty("matrixRowMajor");
      expect(pillow).toMatchObject({
        semanticGranularity: "independent_child_asset",
        parentObjectId: "sam3_bed_01",
        movesWithParent: true,
      });
    }
    const bed = objects.get("sam3_bed_01");
    expect(bed.placement).toMatchObject({
      assetCoordinatesBaked: false,
      matrixRowMajor: BED_MATRIX_ROW_MAJOR,
      pivot: BED_MATRIX_ROW_MAJOR.slice(0, 3).map((row) => row[3]),
    });
    expect(new Set(bed.childObjectIds)).toEqual(new Set(PILLOW_IDS));
    const bedAxisScales = [0, 1, 2].map((column) => Math.hypot(
      bed.placement.matrixRowMajor[0][column],
      bed.placement.matrixRowMajor[1][column],
      bed.placement.matrixRowMajor[2][column],
    ));
    expect(new Set(bedAxisScales.map((scale) => scale.toFixed(6))).size).toBe(3);

    const urls = [...new Set(allWorldUrls(manifest, OUTPUT_URL_PREFIX))];
    expect(urls.length).toBeGreaterThan(20);
    for (const url of urls) {
      expect(fs.statSync(localUrlPath(url, OUTPUT_URL_PREFIX, fixture.outputWorld)).isFile()).toBe(true);
    }
    expect(fs.readFileSync(path.join(fixture.outputWorld, "qa", "browser-qa.json"), "utf8"))
      .toBe("{}\n");
    expect(fs.statSync(path.join(fixture.outputWorld, "baseline-marker.txt")).ino)
      .toBe(fs.statSync(path.join(fixture.sourceWorld, "baseline-marker.txt")).ino);

    const report = JSON.parse(
      fs.readFileSync(path.join(fixture.outputWorld, "qa", "strict-clean-scene-adoption-report.json")),
    );
    expect(report.runtimeAssetResolution).toMatchObject({
      status: "passed",
      allManifestWorldUrlsResolve: true,
      runtimeAssetHashesRevalidated: true,
    });
    expect(report.publicManifest).toEqual({
      sha256Before: fixture.publicSha,
      sha256After: fixture.publicSha,
      unchanged: true,
    });
    expect(report.gates).toMatchObject({
      onlyLegacyFourCarved: true,
      pillowsAndBedNotCarvedAgain: true,
      fourUnifiedPbrAssetsMaterialized: true,
      unifiedPlacementAndHierarchyContractValidated: true,
      publicManifestUnchanged: true,
    });
  });

  test.each([
    ["requires baked pillow coordinates", (pillow) => { pillow.placement.assetCoordinatesBaked = false; }, /assetCoordinatesBaked must be true/u],
    ["forbids pillow matrixRowMajor", (pillow) => { pillow.placement.matrixRowMajor = BED_MATRIX_ROW_MAJOR.map((row) => [...row]); }, /matrixRowMajor is forbidden/u],
    ["requires identity pillow runtime scale", (pillow) => { pillow.placement.scale = [1, 1.01, 1]; }, /placement\.scale must equal \[1, 1, 1\]/u],
    ["requires zero pillow runtime rotation", (pillow) => { pillow.placement.rotationEulerDeg = [0, 1, 0]; }, /rotationEulerDeg must equal \[0, 0, 0\]/u],
    ["requires independent child semantics", (pillow) => { pillow.semanticGranularity = "independent_root_asset"; }, /semantic hierarchy role is invalid/u],
    ["requires the bed parent id", (pillow) => { pillow.parentObjectId = null; }, /parentObjectId is invalid/u],
    ["requires pillows to move with the bed", (pillow) => { pillow.movesWithParent = false; }, /movesWithParent is invalid/u],
  ])("rejects a pillow that %s", (_label, mutatePillow, message) => {
    const fixture = makeFixture();
    mutateCandidate(fixture, (objects) => mutatePillow(objects.get("sam3_pillow_front")));

    expect(() => materialize(fixture)).toThrow(message);
    expect(fs.existsSync(fixture.outputWorld)).toBe(false);
  });

  test.each([
    ["requires unbaked bed coordinates", (bed) => { bed.placement.assetCoordinatesBaked = true; }, /assetCoordinatesBaked must be false/u],
    ["requires a 4x4 bed matrix", (bed) => { bed.placement.matrixRowMajor = bed.placement.matrixRowMajor.slice(0, 3); }, /matrixRowMajor/u],
    ["requires finite bed matrix entries", (bed) => { bed.placement.matrixRowMajor[0][0] = null; }, /matrixRowMajor/u],
    ["requires an affine bed matrix last row", (bed) => { bed.placement.matrixRowMajor[3] = [0, 0, 0, 2]; }, /last row must equal \[0, 0, 0, 1\]/u],
    ["requires bed pivot to equal matrix translation", (bed) => { bed.placement.pivot[0] += 0.001; }, /pivot must equal the matrix translation/u],
    ["requires the exact pillow child set", (bed) => { bed.childObjectIds = PILLOW_IDS.slice(0, 2); }, /childObjectIds must contain exactly/u],
  ])("rejects a bed that %s", (_label, mutateBed, message) => {
    const fixture = makeFixture();
    mutateCandidate(fixture, (objects) => mutateBed(objects.get("sam3_bed_01")));

    expect(() => materialize(fixture)).toThrow(message);
    expect(fs.existsSync(fixture.outputWorld)).toBe(false);
  });

  test("rejects a non-identity alignment even when its receipt hash is refreshed", () => {
    const fixture = makeFixture();
    const receipt = JSON.parse(fs.readFileSync(fixture.alignmentReceipt));
    receipt.transform.translation = [0.01, 0, 0];
    writeJson(fixture.alignmentReceipt, receipt);
    fixture.alignmentReceiptSha = sha256File(fixture.alignmentReceipt);

    expect(() => materialize(fixture)).toThrow(/non-identity alignment translation/u);
    expect(fs.existsSync(fixture.outputWorld)).toBe(false);
    expect(sha256File(fixture.publicManifestPath)).toBe(fixture.publicSha);
  });

  test("rejects a stale alignment receipt hash before reading geometry", () => {
    const fixture = makeFixture();
    expect(() => materialize(fixture, {
      alignmentReceiptSha256: sha256(Buffer.from("stale-receipt")),
    })).toThrow(/alignment receipt: SHA-256 mismatch/u);
    expect(fs.existsSync(fixture.outputWorld)).toBe(false);
  });

  test("rejects a non-watertight bed or missing hierarchy summary", () => {
    const badBed = makeFixture();
    const badBedManifest = JSON.parse(fs.readFileSync(badBed.candidateManifest));
    const bed = badBedManifest.interactiveObjects.find((item) => item.id === "sam3_bed_01");
    bed.collision.topology = "surface_bvh";
    bed.collision.asset.watertight = false;
    writeJson(badBed.candidateManifest, badBedManifest);
    expect(() => materialize(badBed)).toThrow(/collision topology is invalid/u);
    expect(fs.existsSync(badBed.outputWorld)).toBe(false);

    const missingHierarchy = makeFixture();
    const missingManifest = JSON.parse(fs.readFileSync(missingHierarchy.candidateManifest));
    delete missingManifest.candidateBuild.unifiedPbrObjects.hierarchy;
    writeJson(missingHierarchy.candidateManifest, missingManifest);
    expect(() => materialize(missingHierarchy)).toThrow(/object hierarchy is invalid/u);
    expect(fs.existsSync(missingHierarchy.outputWorld)).toBe(false);
  });

  test("rejects post-receipt PGSR tampering and never creates a partial world", () => {
    const fixture = makeFixture();
    fs.appendFileSync(fixture.pgsrPly, "tamper");

    expect(() => materialize(fixture)).toThrow(/alignment input pgsr_ply: SHA-256 mismatch/u);
    expect(fs.existsSync(fixture.outputWorld)).toBe(false);
    expect(sha256File(fixture.publicManifestPath)).toBe(fixture.publicSha);
  });

  test("refuses to overwrite source-world-dir", () => {
    const fixture = makeFixture();
    expect(() => materialize(fixture, { outputWorld: fixture.sourceWorld }))
      .toThrow(/must not overwrite source-world-dir/u);
    expect(sha256File(fixture.publicManifestPath)).toBe(fixture.publicSha);
  });
});
