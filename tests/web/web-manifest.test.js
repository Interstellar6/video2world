import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  placementMatrixElements,
  validateWebManifest,
} from "../../web/web-manifest.js";

const fixturePath = path.resolve("web/public/test-fixtures/manifest.json");

function rawFixture() {
  return JSON.parse(fs.readFileSync(fixturePath, "utf8"));
}

function fixture() {
  const manifest = rawFixture();
  manifest.interactiveObjects[0].collision.gate = {
    status: "passed",
    reason: "runtime test GLB is explicitly approved",
  };
  return manifest;
}

function makeUnifiedObject(object, { topology = "surface_bvh", watertight = false } = {}) {
  const asset = structuredClone(object.collision.asset);
  Object.assign(asset, {
    faces: 1,
    finite: true,
    nondegenerate: true,
    windingConsistent: true,
    watertight,
  });
  object.collision = {
    mode: "unified-glb",
    topology,
    walkable: false,
    characterCollision: true,
    asset,
    gate: {
      status: "passed",
      surfaceCollision: "passed",
      reason: "fixture unified GLB passed surface collision QA",
    },
  };
  delete object.visual;
  delete object.colliderProxy;
  return object;
}

describe("Web manifest contract", () => {
  it("accepts the runtime fixture once its GLB collider has an explicit passed gate", () => {
    const manifest = validateWebManifest(fixture());
    expect(manifest.version).toBe("browser-fixture-v1");
    expect(manifest.interactiveObjects[0]).toMatchObject({
      semanticGranularity: "independent_root_asset",
      parentObjectId: null,
      movesWithParent: false,
      independentlyMovable: true,
    });
  });

  it("rejects an unknown schema version", () => {
    const manifest = fixture();
    manifest.schemaVersion = 2;
    expect(() => validateWebManifest(manifest)).toThrow("schemaVersion must equal 1");
  });

  it("rejects duplicate object identities", () => {
    const manifest = fixture();
    manifest.interactiveObjects.push(structuredClone(manifest.interactiveObjects[0]));
    expect(() => validateWebManifest(manifest)).toThrow("duplicates");
  });

  it("rejects a collider on an explicit visual-only object", () => {
    const manifest = fixture();
    const pillow = manifest.interactiveObjects.find((item) => item.id === "sam3_pillow_01");
    pillow.collision.asset = structuredClone(manifest.interactiveObjects[0].collision.asset);
    expect(() => validateWebManifest(manifest)).toThrow("must not declare collision.asset");
  });

  it("accepts only an explicitly marked non-loadable baseline collider", () => {
    const manifest = fixture();
    delete manifest.assets.collider.url;
    manifest.assets.collider.parts = [];
    manifest.assets.collider.baselineMetadataOnly = true;
    expect(validateWebManifest(manifest).assets.collider.baselineMetadataOnly).toBe(true);

    delete manifest.assets.collider.baselineMetadataOnly;
    expect(() => validateWebManifest(manifest)).toThrow("explicit metadata-only record");
  });

  it("accepts a movable child whose parent is another interactive asset", () => {
    const manifest = fixture();
    const pillow = manifest.interactiveObjects.find((item) => item.id === "sam3_pillow_01");
    Object.assign(pillow, {
      semanticGranularity: "independent_child_asset",
      parentObjectId: "sam3_plant_01",
      movesWithParent: true,
      independentlyMovable: true,
    });

    expect(validateWebManifest(manifest).interactiveObjects.at(-1)).toMatchObject({
      parentObjectId: "sam3_plant_01",
      movesWithParent: true,
    });
  });

  it("accepts a stripped logical hierarchy ancestor without runtime assets", () => {
    const manifest = fixture();
    const ancestor = manifest.interactiveObjects[0];
    const child = manifest.interactiveObjects[2];
    for (const field of ["placement", "colliderProxy", "visual", "collision", "interaction"]) {
      delete ancestor[field];
    }
    Object.assign(ancestor, {
      semanticGranularity: "independent_root_asset",
      parentObjectId: null,
      movesWithParent: false,
      independentlyMovable: false,
      logicalHierarchyOnly: true,
      logicalRole: "unrendered_unselectable_hierarchy_ancestor",
      childObjectIds: [child.id],
    });
    Object.assign(child, {
      semanticGranularity: "independent_child_asset",
      parentObjectId: ancestor.id,
      movesWithParent: true,
      independentlyMovable: true,
      childObjectIds: [],
    });

    const validated = validateWebManifest(manifest);
    expect(validated.interactiveObjects[0]).toMatchObject({
      logicalHierarchyOnly: true,
      independentlyMovable: false,
      childObjectIds: [child.id],
    });
    expect(validated.interactiveObjects[0]).not.toHaveProperty("placement");
    expect(validated.interactiveObjects[0]).not.toHaveProperty("collision");
  });

  it("rejects logical hierarchy ancestors that retain runtime asset fields", () => {
    const manifest = fixture();
    Object.assign(manifest.interactiveObjects[0], {
      independentlyMovable: false,
      logicalHierarchyOnly: true,
      logicalRole: "unrendered_unselectable_hierarchy_ancestor",
      childObjectIds: [],
    });
    expect(() => validateWebManifest(manifest)).toThrow(
      "logical hierarchy nodes must not declare placement",
    );
  });

  it("rejects logical hierarchy ancestors without filtered child metadata", () => {
    const manifest = fixture();
    const ancestor = manifest.interactiveObjects[0];
    for (const field of ["placement", "colliderProxy", "visual", "collision", "interaction"]) {
      delete ancestor[field];
    }
    Object.assign(ancestor, {
      independentlyMovable: false,
      logicalHierarchyOnly: true,
      logicalRole: "unrendered_unselectable_hierarchy_ancestor",
    });
    expect(() => validateWebManifest(manifest)).toThrow(
      "childObjectIds is required for logical hierarchy nodes",
    );
  });

  it("rejects logical hierarchy child metadata that disagrees with parent ids", () => {
    const manifest = fixture();
    const ancestor = manifest.interactiveObjects[0];
    const child = manifest.interactiveObjects[2];
    for (const field of ["placement", "colliderProxy", "visual", "collision", "interaction"]) {
      delete ancestor[field];
    }
    Object.assign(ancestor, {
      independentlyMovable: false,
      logicalHierarchyOnly: true,
      logicalRole: "unrendered_unselectable_hierarchy_ancestor",
      childObjectIds: [manifest.interactiveObjects[1].id],
    });
    Object.assign(child, {
      semanticGranularity: "independent_child_asset",
      parentObjectId: ancestor.id,
      movesWithParent: true,
    });

    expect(() => validateWebManifest(manifest)).toThrow(
      "childObjectIds must exactly match parentObjectId relationships",
    );
  });

  it("rejects incomplete child declarations and unknown parents", () => {
    const missingParent = fixture();
    Object.assign(missingParent.interactiveObjects[2], {
      semanticGranularity: "independent_child_asset",
      movesWithParent: true,
    });
    expect(() => validateWebManifest(missingParent)).toThrow(
      "independent_child_asset and parentObjectId must be declared together",
    );

    const unknownParent = fixture();
    Object.assign(unknownParent.interactiveObjects[2], {
      semanticGranularity: "independent_child_asset",
      parentObjectId: "missing_bed",
      movesWithParent: true,
    });
    expect(() => validateWebManifest(unknownParent)).toThrow("references unknown object missing_bed");
  });

  it("rejects invalid movement semantics and merged movable fragments", () => {
    const childWithoutInheritance = fixture();
    Object.assign(childWithoutInheritance.interactiveObjects[2], {
      semanticGranularity: "independent_child_asset",
      parentObjectId: "sam3_plant_01",
      movesWithParent: false,
    });
    expect(() => validateWebManifest(childWithoutInheritance)).toThrow(
      "movesWithParent must be true exactly for child assets",
    );

    const movableFragment = fixture();
    Object.assign(movableFragment.interactiveObjects[2], {
      semanticGranularity: "merged_component",
      independentlyMovable: true,
    });
    expect(() => validateWebManifest(movableFragment)).toThrow(
      "merged components cannot be independently movable",
    );
  });

  it("rejects hierarchy cycles", () => {
    const manifest = fixture();
    Object.assign(manifest.interactiveObjects[0], {
      semanticGranularity: "independent_child_asset",
      parentObjectId: "sam3_plant_02",
      movesWithParent: true,
    });
    Object.assign(manifest.interactiveObjects[1], {
      semanticGranularity: "independent_child_asset",
      parentObjectId: "sam3_plant_01",
      movesWithParent: true,
    });

    expect(() => validateWebManifest(manifest)).toThrow("hierarchy contains a cycle");
  });

  it("accepts a strictly scoped optional scene command service", () => {
    const manifest = fixture();
    manifest.sceneCommandService = {
      contract: "video2world-scene-command-service-1.0.0",
      endpoint: "/v1/scene-commands/submit",
    };
    expect(validateWebManifest(manifest).sceneCommandService.endpoint).toBe(
      "/v1/scene-commands/submit",
    );
  });

  it("rejects unsafe or loosely shaped scene command endpoints", () => {
    const insecure = fixture();
    insecure.sceneCommandService = { endpoint: "http://example.com/v1/scene-commands/submit" };
    expect(() => validateWebManifest(insecure)).toThrow("must use HTTPS outside localhost");

    const credentialed = fixture();
    credentialed.sceneCommandService = { endpoint: "https://user:secret@example.com/command" };
    expect(() => validateWebManifest(credentialed)).toThrow("must not contain credentials");

    const unknown = fixture();
    unknown.sceneCommandService = { endpoint: "/v1/scene-commands/submit", headers: { Authorization: "secret" } };
    expect(() => validateWebManifest(unknown)).toThrow("unknown fields");
  });

  it("validates an optional canonical source manifest hash for optimistic locking", () => {
    const manifest = fixture();
    manifest.sourceWorld = {
      worldId: "bedroom4",
      runId: "fixture-run",
      adoptionMode: "verified",
      manifestSha256: "a".repeat(64),
    };
    expect(validateWebManifest(manifest).sourceWorld.manifestSha256).toBe("a".repeat(64));

    manifest.sourceWorld.manifestSha256 = "A".repeat(64);
    expect(() => validateWebManifest(manifest)).toThrow("64 lowercase hexadecimal");
  });

  it("validates only the interaction modes implemented by the runtime", () => {
    const manifest = fixture();
    expect(validateWebManifest(manifest).interactiveObjects[2].interaction).toMatchObject({
      kind: "spin",
      drag: "horizontal_yaw",
    });

    const unsupportedKind = fixture();
    unsupportedKind.interactiveObjects[0].interaction.kind = "translate";
    expect(() => validateWebManifest(unsupportedKind)).toThrow("interaction.kind is unsupported");

    const unsupportedDrag = fixture();
    unsupportedDrag.interactiveObjects[0].interaction.drag = "free_xyz";
    expect(() => validateWebManifest(unsupportedDrag)).toThrow("interaction.drag is unsupported");

    const partialSpin = fixture();
    partialSpin.interactiveObjects[0].interaction.degrees = 180;
    expect(() => validateWebManifest(partialSpin)).toThrow("interaction.degrees must equal 360");

    const invalidDuration = fixture();
    invalidDuration.interactiveObjects[0].interaction.durationMs = 0;
    expect(() => validateWebManifest(invalidDuration)).toThrow(
      "interaction.durationMs must be a positive finite number",
    );
  });

  it("treats a missing interaction as selection-only instead of granting movement", () => {
    const manifest = fixture();
    delete manifest.interactiveObjects[0].interaction;
    expect(validateWebManifest(manifest).interactiveObjects[0].interaction).toBeUndefined();
  });

  it("requires an explicit, recognized collision gate for every collidable object", () => {
    const missingGate = rawFixture();
    expect(() => validateWebManifest(missingGate)).toThrow(
      "collision.gate is required for collidable objects",
    );

    const unknownGate = fixture();
    unknownGate.interactiveObjects[0].collision.gate.status = "maybe";
    expect(() => validateWebManifest(unknownGate)).toThrow(
      "collision.gate.status is unsupported",
    );

    const heldCandidate = fixture();
    heldCandidate.interactiveObjects[0].collision.gate.status = "held";
    expect(validateWebManifest(heldCandidate).interactiveObjects[0].collision.gate.status)
      .toBe("held");
  });

  it("requires a positive collider proxy so focus and fallback geometry cannot be implicit", () => {
    const missing = fixture();
    delete missing.interactiveObjects[0].colliderProxy;
    expect(() => validateWebManifest(missing)).toThrow("colliderProxy must be an object");

    const zeroExtent = fixture();
    zeroExtent.interactiveObjects[0].colliderProxy.dimensions[1] = 0;
    expect(() => validateWebManifest(zeroExtent)).toThrow(
      "colliderProxy.dimensions must contain three positive finite numbers",
    );

    const unsupported = fixture();
    unsupported.interactiveObjects[0].colliderProxy.type = "capsule";
    expect(() => validateWebManifest(unsupported)).toThrow("colliderProxy.type is unsupported");
  });

  it("accepts only the Y-up axes implemented by robot, drag, and camera physics", () => {
    const positiveY = fixture();
    positiveY.coordinateSystem.worldUp = [0, 1, 0];
    expect(validateWebManifest(positiveY).coordinateSystem.worldUp).toEqual([0, 1, 0]);

    const xUp = fixture();
    xUp.coordinateSystem.worldUp = [1, 0, 0];
    expect(() => validateWebManifest(xUp)).toThrow(
      "coordinateSystem.worldUp must be [0, 1, 0] or [0, -1, 0]",
    );

    const tilted = fixture();
    tilted.coordinateSystem.worldUp = [0, 0.99, 0.1];
    expect(() => validateWebManifest(tilted)).toThrow(
      "coordinateSystem.worldUp must be [0, 1, 0] or [0, -1, 0]",
    );
  });

  it("accepts one explicit surface-BVH GLB as visual, selection, and collision geometry", () => {
    const manifest = fixture();
    makeUnifiedObject(manifest.interactiveObjects[0]);

    const unified = validateWebManifest(manifest).interactiveObjects[0];
    expect("visual" in unified).toBe(false);
    expect("colliderProxy" in unified).toBe(false);
    expect(unified).toMatchObject({
      collision: {
        mode: "unified-glb",
        topology: "surface_bvh",
        asset: {
          fileType: "glb",
          watertight: false,
        },
        gate: { status: "passed", surfaceCollision: "passed" },
      },
    });
  });

  it("accepts a decomposable row-major 4x4 for an unbaked unified GLB", () => {
    const manifest = fixture();
    const object = makeUnifiedObject(manifest.interactiveObjects[0], {
      topology: "closed_volume",
      watertight: true,
    });
    object.placement = {
      ...object.placement,
      generatedCenter: [0, 0, 0],
      pivot: [1, 2, 3],
      scale: [1, 1, 1],
      rotationEulerDeg: [0, 0, 0],
      matrixRowMajor: [
        [0, -3, 0, 1],
        [2, 0, 0, 2],
        [0, 0, 4, 3],
        [0, 0, 0, 1],
      ],
    };
    const validated = validateWebManifest(manifest).interactiveObjects[0];
    expect(placementMatrixElements(validated.placement)).toEqual([
      0, 2, 0, 0,
      -3, 0, 0, 0,
      0, 0, 4, 0,
      1, 2, 3, 1,
    ]);
  });

  it("rejects a sheared matrix or a pivot that disagrees with matrix translation", () => {
    const sheared = fixture();
    const shearedObject = makeUnifiedObject(sheared.interactiveObjects[0]);
    Object.assign(shearedObject.placement, {
      generatedCenter: [0, 0, 0],
      pivot: [1, 2, 3],
      scale: [1, 1, 1],
      rotationEulerDeg: [0, 0, 0],
      matrixRowMajor: [
        [1, 0.25, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 0, 1],
      ],
    });
    expect(() => validateWebManifest(sheared)).toThrow(
      "contains shear and cannot be decomposed for root interaction",
    );

    const wrongPivot = fixture();
    const wrongPivotObject = makeUnifiedObject(wrongPivot.interactiveObjects[0]);
    Object.assign(wrongPivotObject.placement, {
      generatedCenter: [0, 0, 0],
      pivot: [0, 0, 0],
      scale: [1, 1, 1],
      rotationEulerDeg: [0, 0, 0],
      matrixRowMajor: [
        [1, 0, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 0, 1],
      ],
    });
    expect(() => validateWebManifest(wrongPivot)).toThrow(
      "placement.pivot must equal the matrix translation",
    );
  });

  it("requires an explicit unified mode instead of promoting collision assets implicitly", () => {
    const missingVisual = fixture();
    delete missingVisual.interactiveObjects[0].visual;
    expect(() => validateWebManifest(missingVisual)).toThrow("visual must be an object");

    const separateRenderAsset = fixture();
    separateRenderAsset.interactiveObjects[0].collision.renderAsset = structuredClone(
      separateRenderAsset.interactiveObjects[0].collision.asset,
    );
    delete separateRenderAsset.interactiveObjects[0].visual;
    expect(() => validateWebManifest(separateRenderAsset)).toThrow("visual must be an object");
  });

  it("fails closed on unsafe unified topology, density, closure, or gate evidence", () => {
    const unbakedOrigin = fixture();
    makeUnifiedObject(unbakedOrigin.interactiveObjects[0]);
    unbakedOrigin.interactiveObjects[0].placement.generatedCenter = [1, 0, 0];
    expect(() => validateWebManifest(unbakedOrigin)).toThrow(
      "requires a baked local origin at generatedCenter [0, 0, 0]",
    );

    const tooDense = fixture();
    makeUnifiedObject(tooDense.interactiveObjects[0]).collision.asset.faces = 100_001;
    expect(() => validateWebManifest(tooDense)).toThrow(
      "collision.asset.faces must be between 1 and 100000",
    );

    const degenerate = fixture();
    makeUnifiedObject(degenerate.interactiveObjects[0]).collision.asset.nondegenerate = false;
    expect(() => validateWebManifest(degenerate)).toThrow(
      "collision.asset must be finite, nondegenerate, and winding-consistent",
    );

    const openVolume = fixture();
    makeUnifiedObject(openVolume.interactiveObjects[0], {
      topology: "closed_volume",
      watertight: false,
    });
    expect(() => validateWebManifest(openVolume)).toThrow(
      "closed_volume requires a watertight GLB",
    );

    const failedGate = fixture();
    makeUnifiedObject(failedGate.interactiveObjects[0]).collision.gate.surfaceCollision = "failed";
    expect(() => validateWebManifest(failedGate)).toThrow(
      "requires passed gate.status and gate.surfaceCollision",
    );
  });
});
