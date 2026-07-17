import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

import {
  placementMatrixElements,
  validateWebManifest,
} from "../../web/web-manifest.js";

const root = path.resolve(".");
const manifestPath = path.join(
  root,
  "examples/bedroom4/manifests/bedroom4.unified-pbr-objects.candidate.web.json",
);
const reportPath = path.join(
  root,
  "examples/bedroom4/manifests/bedroom4.unified-pbr-objects.candidate.report.json",
);
const basePath = path.join(
  root,
  "examples/bedroom4/manifests/bedroom4.unified-pillow.candidate.web.json",
);
const objectIds = [
  "sam3_pillow_front",
  "sam3_pillow_left",
  "sam3_pillow_right",
  "sam3_bed_01",
];

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function digest(file) {
  return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

function sourceFile(sourcePath) {
  const prefix = "repo://video2world/";
  expect(sourcePath.startsWith(prefix)).toBe(true);
  return path.join(root, sourcePath.slice(prefix.length));
}

describe("Bedroom 4 four-object unified PBR candidate", () => {
  it("validates four unified GLBs without visual or collider proxy splits", () => {
    const manifest = readJson(manifestPath);
    expect(() => validateWebManifest(structuredClone(manifest))).not.toThrow();
    const objects = manifest.interactiveObjects.filter((item) => objectIds.includes(item.id));
    expect(objects.map((item) => item.id)).toEqual(objectIds);
    objects.forEach((object) => {
      expect(object).not.toHaveProperty("visual");
      expect(object).not.toHaveProperty("renderAsset");
      expect(object).not.toHaveProperty("colliderProxy");
      expect(object.collision).not.toHaveProperty("renderAsset");
      expect(object.collision).toMatchObject({
        mode: "unified-glb",
        characterCollision: true,
        gate: {
          status: "passed",
          surfaceCollision: "passed",
          candidateBrowserQa: "pending",
        },
      });
      expect(object.interaction).toEqual({
        kind: "spin",
        degrees: 360,
        durationMs: 1250,
        drag: "horizontal_yaw",
      });
      expect(object.independentlyMovable).toBe(true);
      const file = sourceFile(object.collision.asset.sourcePath);
      expect(fs.statSync(file).size).toBe(object.collision.asset.size);
      expect(digest(file)).toBe(object.collision.asset.sha256);
    });
    const byId = new Map(objects.map((object) => [object.id, object]));
    objectIds.slice(0, 3).forEach((objectId) => {
      expect(byId.get(objectId)).toMatchObject({
        semanticGranularity: "independent_child_asset",
        parentObjectId: "sam3_bed_01",
        movesWithParent: true,
        independentlyMovable: true,
      });
    });
    expect(byId.get("sam3_bed_01")).toMatchObject({
      semanticGranularity: "independent_root_asset",
      parentObjectId: null,
      movesWithParent: false,
      independentlyMovable: true,
    });
  });

  it("keeps baked pillow pivots separate from the unbaked bed 4x4 transform", () => {
    const manifest = readJson(manifestPath);
    const objects = new Map(manifest.interactiveObjects.map((item) => [item.id, item]));
    objectIds.slice(0, 3).forEach((objectId) => {
      const pillow = objects.get(objectId);
      expect(pillow.placement.assetCoordinatesBaked).toBe(true);
      expect(pillow.placement).not.toHaveProperty("matrixRowMajor");
      expect(pillow.placement.scale).toEqual([1, 1, 1]);
      expect(pillow.placement.generatedCenter).toEqual([0, 0, 0]);
      expect(pillow.collision.topology).toBe("surface_bvh");
      expect(pillow.collision.asset.watertight).toBe(false);
    });

    const bed = objects.get("sam3_bed_01");
    expect(bed.placement.assetCoordinatesBaked).toBe(false);
    expect(bed.placement.matrixRowMajor).toHaveLength(4);
    expect(bed.placement.pivot).toEqual(
      bed.placement.matrixRowMajor.slice(0, 3).map((row) => row[3]),
    );
    expect(bed.collision).toMatchObject({
      topology: "closed_volume",
      asset: {
        logicalRoot: "sam3_bed_01",
        vertices: 2500,
        faces: 4976,
        watertight: true,
      },
    });
    expect(placementMatrixElements(bed.placement)).toEqual([
      6.0940405667591,
      -0.20072433617157628,
      1.9936905918532148,
      0,
      0.06831981888678311,
      -12.444750282849224,
      -1.4617654309371404,
      0,
      2.648131223143048,
      0.9540323820267776,
      -7.998393355783393,
      0,
      0.639241276163562,
      5.6921053762733935,
      10.762354974571135,
      1,
    ]);
  });

  it("reuses static assets exactly and remains blocked on clean-scene reconstruction", () => {
    const base = readJson(basePath);
    const manifest = readJson(manifestPath);
    const report = readJson(reportPath);
    expect(manifest.assets).toEqual(base.assets);
    expect(manifest.candidateBuild).toMatchObject({
      status: "candidate_materialized_clean_scene_pending",
      promotionAllowed: false,
      cleanScene: {
        status: "clean_scene_pending",
        methodRequired: "front_to_back_cumulative_peel_then_clean_plate_reconstruction",
      },
    });
    expect(report).toMatchObject({
      status: "candidate_materialized_clean_scene_pending",
      promotion_allowed: false,
      public_manifests: {
        mutated: false,
        promoted_manifest_sha256:
          "7e0a1dcd801225bf21bd5a77e82edcce2318e348523d69ca80b7d1e5b53dc96d",
        stable_manifest_sha256:
          "3805f0e5bab09add424b3b78f9349cd2eca6d1262777ef683e13cda07695e82b",
      },
    });
    expect(digest(manifestPath)).toBe(report.output_manifest.sha256);
  });
});
