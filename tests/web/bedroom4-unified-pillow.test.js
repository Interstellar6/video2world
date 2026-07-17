import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { validateWebManifest } from "../../web/web-manifest.js";
import { buildSceneKnowledgeIndex, resolveSceneEntity } from "../../web/scene-query.js";

const root = path.resolve(".");
const manifestPath = path.join(
  root,
  "examples/bedroom4/manifests/bedroom4.unified-pillow.candidate.web.json",
);

function fixture() {
  return JSON.parse(fs.readFileSync(manifestPath, "utf8"));
}

function digest(file) {
  return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

function nestedUrls(value, result = []) {
  if (Array.isArray(value)) {
    value.forEach((item) => nestedUrls(item, result));
  } else if (value && typeof value === "object") {
    Object.entries(value).forEach(([key, item]) => {
      if (key === "url") result.push(item);
      else nestedUrls(item, result);
    });
  }
  return result;
}

function plyVertexCount(file) {
  const bytes = fs.readFileSync(file);
  const end = bytes.indexOf(Buffer.from("end_header\n"));
  expect(end).toBeGreaterThan(0);
  const header = bytes.subarray(0, end).toString("ascii");
  return Number(header.match(/^element vertex (\d+)$/mu)?.[1]);
}

describe("Bedroom 4 unified pillow candidate", () => {
  it("passes the Web manifest contract with one explicit unified front GLB", () => {
    const manifest = validateWebManifest(fixture());
    const front = manifest.interactiveObjects.find((item) => item.id === "sam3_pillow_front");

    expect(front).toBeTruthy();
    expect(front.aliases.slice(0, 2)).toEqual(["pillow", "枕头"]);
    expect(nestedUrls(front)).toEqual([
      "./worlds/bedroom4/objects/sam3_pillow_front.scene-fit.glb",
    ]);
    expect(front).not.toHaveProperty("visual");
    expect(front).not.toHaveProperty("renderAsset");
    expect(front).not.toHaveProperty("colliderProxy");
    expect(front.collision).not.toHaveProperty("renderAsset");
    expect(front).toMatchObject({
      placement: {
        pivot: [-0.9132125483707731, -0.1424510127463641, 14.760714277636716],
        generatedCenter: [0, 0, 0],
        scale: [1, 1, 1],
        rotationEulerDeg: [0, 0, 0],
      },
      collision: {
        mode: "unified-glb",
        topology: "surface_bvh",
        walkable: false,
        characterCollision: true,
        asset: {
          vertices: 60237,
          faces: 97082,
          finite: true,
          nondegenerate: true,
          windingConsistent: true,
          watertight: false,
        },
        gate: {
          status: "passed",
          surfaceCollision: "passed",
          rootProxyBrowserQa: "pending_promoted_manifest_recheck",
        },
      },
    });
    const glb = path.join(
      root,
      "examples/bedroom4/completion/trellis2_pillow_front_seed42/",
      "scene_fit_silhouette_refined/sam3_pillow_front.scene-fit.glb",
    );
    expect(fs.existsSync(glb)).toBe(true);
    expect(fs.statSync(glb).size).toBe(front.collision.asset.size);
    expect(digest(glb)).toBe(front.collision.asset.sha256);

    const index = buildSceneKnowledgeIndex(manifest);
    expect(resolveSceneEntity("枕头在哪里", index).entity.id).toBe("sam3_pillow_front");
    expect(resolveSceneEntity("pillow appearance", index).entity.id).toBe("sam3_pillow_front");
    expect(resolveSceneEntity("左侧后排枕头", index).entity.id).toBe("sam3_pillow_left");
    expect(resolveSceneEntity("右侧后排枕头", index).entity.id).toBe("sam3_pillow_right");
  });

  it("restores the two rear split RGB surfaces without keeping the old ensemble", () => {
    const manifest = fixture();
    expect(JSON.stringify(manifest)).not.toContain("sam3_pillow_01");
    const interactiveIds = manifest.interactiveObjects.map((item) => item.id);
    const knowledgeIds = manifest.sceneKnowledge.objects.map((item) => item.id);
    expect(interactiveIds).not.toContain("sam3_pillow_01");
    expect(knowledgeIds).not.toContain("sam3_pillow_01");
    expect(manifest.sceneKnowledge.relations).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({ subject: "sam3_pillow_01" }),
      ]),
    );

    const expected = {
      sam3_pillow_left: {
        points: 89763,
        bytes: 1346659,
        sha256: "b8c45454d887a77a99b17558c0fa0f0e49d81ab00099e346aeffc940eacdb400",
      },
      sam3_pillow_right: {
        points: 49163,
        bytes: 737659,
        sha256: "09537489887ad6ba3cbccf9cbaae99142eccbb4a65ab82b8829d4fe140bfc63f",
      },
    };
    for (const [id, contract] of Object.entries(expected)) {
      const object = manifest.interactiveObjects.find((item) => item.id === id);
      const file = path.join(
        root,
        `examples/bedroom4/completion/bedroom4_frame64_three_pillows/objects/${id}/${id}.ply`,
      );
      expect(object).not.toHaveProperty("interaction");
      expect(object).toMatchObject({
        visualOnly: true,
        independentlyMovable: false,
        collision: { mode: "none", characterCollision: false },
        colliderProxy: { type: "selection-box", collisionEnabled: false },
        visual: {
          vertexCount: contract.points,
          size: contract.bytes,
          sha256: contract.sha256,
        },
      });
      expect(fs.existsSync(file)).toBe(true);
      expect(plyVertexCount(file)).toBe(contract.points);
      expect(fs.statSync(file).size).toBe(contract.bytes);
      expect(digest(file)).toBe(contract.sha256);
      expect(object.placement.pivot).toEqual(object.bbox.center);
    }
  });

  it("records user-scoped acceptance without claiming clean 3D background completion", () => {
    const manifest = fixture();
    expect(manifest.version).toContain("candidate");
    expect(manifest.candidateBuild.cleanPlateReview).toMatchObject({
      status: "accepted_for_current_demo",
      selectedSeed: 2026071701,
      scope: "accepted_for_current_demo_only",
    });
    expect(manifest.candidateBuild.backgroundGeometry.status).toBe("not_proven");
    expect(manifest.candidateBuild.sceneColliderCarve).toMatchObject({
      status: "passed",
      retainedIntersectingFaces: 2797,
      protectedSupportFaceCount: 2797,
      retainedUnprotectedIntersectingFaces: 0,
    });
    expect(manifest.candidateBuild.sceneColliderCarve.additionalRemovedFaceCount)
      .toBeGreaterThan(0);
    const collider = manifest.assets.colliderStaticCarved;
    expect(collider).not.toHaveProperty("url");
    expect(collider.transportMode).toBe("verified_chunks");
    expect(collider.directOversizedAssetPublished).toBe(false);
    expect(collider.parts).toHaveLength(34);
    expect(collider.parts.reduce((total, part) => total + part.size, 0)).toBe(collider.size);
    collider.parts.forEach((part, index) => {
      expect(part.size).toBeLessThanOrEqual(1024 * 1024);
      expect(part.sha256).toMatch(/^[0-9a-f]{64}$/u);
      expect(part.url).toBe(
        `./worlds/bedroom4/chunks/collider_bedroom4_tsdf_static_carved_unified_pillow_ply.chunk${String(index).padStart(3, "0")}`,
      );
    });
    expect(manifest.collisionWorld.gate).toMatchObject({
      status: "passed",
      retainedIntersectingFaces: 2797,
      protectedSupportFaceCount: 2797,
      retainedUnprotectedIntersectingFaces: 0,
    });
    expect(manifest.collisionWorld.sceneAssetTransport).toMatchObject({
      mode: "verified_chunks",
      parts: collider.parts,
      chunkSize: 1024 * 1024,
      size: collider.size,
      sha256: collider.sha256,
      directOversizedAssetPublished: false,
    });
    expect(manifest.candidateBuild.primitiveCounts).toEqual({
      staticSceneGaussians: 821785,
      existingObjectGaussians: 480000,
      rearRgbPoints: 138926,
      frontMeshVertices: 60237,
      totalVisualPrimitives: 1500948,
      existingObjectColliderFaces: 67660,
      frontSurfaceColliderFaces: 97082,
      totalObjectColliderFaces: 164742,
      collidableObjects: 5,
    });
    expect(manifest.productionBuild.baselineHistory.status).toBe(
      "passed_for_base_manifest_only",
    );
    expect(manifest.productionBuild.candidatePillowIntegration.rootProxyBrowserQa).toBe(
      "pending_promoted_manifest_recheck",
    );
    expect(manifest.candidateBuild.promotedManifestBrowserQa.status).toBe(
      "pending_promoted_manifest_recheck",
    );
    const visual = manifest.assets.visual;
    expect(visual).toMatchObject({
      sha256: "7f8a7d9a1fc07cee34754469d5fffe95d90640bfa9d5883b4432e83698e55c0d",
      size: 203804211,
      vertexCount: 821785,
      latestRemovedVertexCount: 1606,
      transportMode: "verified_chunks",
      directOversizedAssetPublished: false,
    });
    expect(visual).not.toHaveProperty("url");
    expect(visual.parts).toHaveLength(195);
    expect(visual.parts.reduce((total, part) => total + part.size, 0)).toBe(visual.size);
    visual.parts.forEach((part, index) => {
      expect(part.size).toBeLessThanOrEqual(1024 * 1024);
      expect(part.sha256).toMatch(/^[0-9a-f]{64}$/u);
      expect(part.url).toBe(
        `./worlds/bedroom4/chunks/visual_bedroom4_static_front_recarve_semantic_covariance.chunk${String(index).padStart(3, "0")}`,
      );
    });
    const urls = nestedUrls(manifest).map(String);
    expect(urls.some((url) => url.includes("recarve-candidates"))).toBe(false);
    expect(urls).not.toContain(
      "./worlds/bedroom4/colliders/collider_bedroom4_tsdf_static_carved_unified_pillow.ply",
    );
    expect(manifest.candidateBuild.qualityPolicy.minorAcceptedLimitations).toContain(
      "minor backside floral-pattern difference on the front pillow",
    );
    expect(manifest.candidateBuild.qualityPolicy.blockingConditions).toEqual([
      "obvious deformation",
      "wrong main color",
      "missing surface",
      "significant scene interpenetration",
    ]);
  });
});
