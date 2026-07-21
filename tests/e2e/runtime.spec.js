import fs from "node:fs";
import path from "node:path";
import { expect, test } from "@playwright/test";

const fixtureUrl = "/?visual=off&manifest=/test-fixtures/manifest.json";
const fixturePath = path.resolve("web/public/test-fixtures/manifest.json");

function runtimeFixture() {
  const manifest = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
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

function matrixGram(matrix) {
  const columns = [
    [matrix[0], matrix[1], matrix[2]],
    [matrix[4], matrix[5], matrix[6]],
    [matrix[8], matrix[9], matrix[10]],
  ];
  const dot = (left, right) => left.reduce((sum, value, index) => sum + value * right[index], 0);
  return [
    dot(columns[0], columns[0]),
    dot(columns[1], columns[1]),
    dot(columns[2], columns[2]),
    dot(columns[0], columns[1]),
    dot(columns[0], columns[2]),
    dot(columns[1], columns[2]),
  ];
}

function hierarchyFixture() {
  const manifest = runtimeFixture();
  const parent = manifest.interactiveObjects.find((item) => item.id === "sam3_plant_01");
  makeUnifiedObject(parent);
  const [x, y, z] = parent.placement.pivot;
  parent.placement.matrixRowMajor = [
    [0, -3, 0, x],
    [2, 0, 0, y],
    [0, 0, 4, z],
    [0, 0, 0, 1],
  ];
  const child = manifest.interactiveObjects.find((item) => item.id === "sam3_pillow_01");
  Object.assign(child, {
    semanticGranularity: "independent_child_asset",
    parentObjectId: "sam3_plant_01",
    movesWithParent: true,
    independentlyMovable: true,
  });
  return manifest;
}

function logicalHierarchyFixture() {
  const manifest = runtimeFixture();
  const ancestor = manifest.interactiveObjects.find((item) => item.id === "sam3_plant_01");
  const child = manifest.interactiveObjects.find((item) => item.id === "sam3_pillow_01");
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
  manifest.initialState.cameraFocusObjectId = ancestor.id;
  return manifest;
}

function cameraOverviewFixture() {
  const manifest = runtimeFixture();
  const target = manifest.interactiveObjects.find((item) => item.id === "sam3_plant_01");
  Object.assign(target, {
    label: "Bed-sized overview target",
    category: "bed",
    colliderProxy: {
      type: "box",
      dimensions: [8, 2.2, 6],
      center: [0, 0, 0],
    },
    collision: {
      mode: "degraded-box",
      walkable: false,
      characterCollision: true,
      asset: null,
      gate: {
        status: "passed",
        reason: "deterministic overview framing fixture",
      },
    },
  });
  manifest.initialState.cameraFocusObjectId = target.id;
  return manifest;
}

test.beforeEach(async ({ page }) => {
  await page.route((url) => url.pathname === "/test-fixtures/manifest.json", (route) => {
    return route.fulfill({ json: runtimeFixture() });
  });
});

function expectVectorClose(actual, expected, epsilon = 0.0002) {
  for (const key of ["x", "y", "z"]) {
    expect(Math.abs(actual[key] - expected[key])).toBeLessThan(epsilon);
  }
}

test("fresh-clone root loads the gated runtime fixture without local production assets", async ({ page }) => {
  const consoleProblems = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) consoleProblems.push(`${message.type()}: ${message.text()}`);
  });
  page.on("pageerror", (error) => consoleProblems.push(`pageerror: ${error.message}`));

  await page.goto("/", { waitUntil: "domcontentloaded" });
  await expect(page).toHaveTitle("Video2World · Interactive 3D Scene");
  await page.waitForFunction(() => {
    const state = window.__visualPhysicsDemoState;
    return state?.visualLoadSkipped
      && state?.colliderReady
      && state?.sceneQaReady
      && state?.interactiveObjectReadyCount === 3;
  });
  await expect(page.locator("#modeChip")).toContainText("visual PLY deferred");
  await expect(page.locator("#objectMetric")).toHaveText("3/3");
  expect(consoleProblems).toEqual([]);
});

test("scene QA, object transforms, and collision hooks work together", async ({ page }) => {
  const consoleProblems = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) consoleProblems.push(`${message.type()}: ${message.text()}`);
  });
  page.on("pageerror", (error) => consoleProblems.push(`pageerror: ${error.message}`));

  await page.goto(fixtureUrl, { waitUntil: "domcontentloaded" });
  await expect(page).toHaveTitle("Video2World · Interactive 3D Scene");
  await expect(page.locator("h1")).toContainText("3DGS");
  await page.waitForFunction(() => {
    const state = window.__visualPhysicsDemoState;
    return state?.colliderReady
      && state?.sceneQaReady
      && state?.interactiveObjectCount === 3
      && state?.objectColliderCount === 2
      && state?.objectColliderReadyCount === 2;
  });
  expect(await page.evaluate(() => ({
    preview: typeof window.__previewSceneCommand,
    confirm: typeof window.__confirmSceneCommand,
    cancel: typeof window.__cancelSceneCommand,
  }))).toEqual({ preview: "function", confirm: "function", cancel: "function" });
  const visualObjectState = await page.evaluate(() => window.__loadInteractiveObjectVisuals());
  expect(visualObjectState.interactiveObjectReadyCount).toBe(3);
  expect(visualObjectState.interactiveObjectRgbPointCount).toBe(24);
  expect(visualObjectState.interactiveObjectGaussianCount).toBe(0);
  const baselineState = await page.evaluate(() => window.__setCameraPreset("reference"));
  expectVectorClose(baselineState.cameraPosition, {
    x: 6.131334,
    y: 1.818849,
    z: 27.751021,
  }, 0.00001);
  expect(baselineState.cameraFovDeg).toBe(55.776917);
  expect(baselineState.cameraPresetSource).toBe("pgsr_input_camera_000018");
  expect(baselineState.robotSpawnSource).toBe("fixed-user-approved-interior-20260716");

  const qaInput = page.locator("#sceneQaInput");
  await qaInput.fill("左侧盆栽长什么样？");
  await page.locator("#sceneQaSubmit").click();
  await expect(page.locator("#sceneQaAnswer")).toContainText("绿色阔叶植物");
  await expect.poll(() => page.evaluate(() => window.__visualPhysicsDemoState.selectedSceneEntity)).toBe("sam3_plant_01");

  await qaInput.fill("植物在哪里？");
  await page.locator("#sceneQaSubmit").click();
  await expect(page.locator("#sceneQaCandidates button")).toHaveCount(2);

  await qaInput.fill("枕头在哪里？");
  await page.locator("#sceneQaSubmit").click();
  await expect(page.locator("#sceneQaAnswer")).toContainText("床上");
  await expect(page.locator("#sceneQaAnswer")).toHaveAttribute("data-status", "resolved");
  await expect.poll(() => page.evaluate(() => window.__visualPhysicsDemoState.selectedInteractiveObject))
    .toBe("sam3_pillow_01");

  await qaInput.fill("删除左侧盆栽");
  await page.locator("#sceneQaSubmit").click();
  await expect(page.locator("#sceneCommandPreview")).toBeVisible();
  await expect(page.locator("#sceneCommandTitle")).toContainText("删除物体");
  await expect(page.locator("#sceneCommandStatus")).toContainText("未配置执行后端");
  await expect(page.locator("#sceneCommandConfirm")).toBeDisabled();
  expect(await page.evaluate(() => window.__visualPhysicsDemoState.sceneCommandTargetIds))
    .toEqual(["sam3_plant_01"]);
  await page.locator("#sceneCommandCancel").click();
  await expect(page.locator("#sceneCommandPreview")).toBeHidden();

  const robotBefore = await page.evaluate(() => window.__visualPhysicsDemoState.robotPosition);
  await qaInput.focus();
  await qaInput.pressSequentially("wasd ");
  await page.waitForTimeout(160);
  const robotAfter = await page.evaluate(() => window.__visualPhysicsDemoState.robotPosition);
  expect(robotAfter).toEqual(robotBefore);

  const collisionWorld = await page.evaluate(() => window.__inspectCollisionWorld());
  expect(collisionWorld.sceneReady).toBe(true);
  expect(collisionWorld.objectColliderReadyCount).toBe(2);
  expect(collisionWorld.degradedCount).toBe(1);
  expect(collisionWorld.targetCount).toBe(3);
  expect(collisionWorld.objects.find((item) => item.id === "sam3_plant_01")).toMatchObject({
    mode: "glb",
    gateStatus: "passed",
    walkable: true,
    characterCollision: true,
    faces: 1,
  });
  expect(collisionWorld.objects.find((item) => item.id === "sam3_plant_02")).toMatchObject({
    mode: "degraded-box",
    gateStatus: "not_tested",
    walkable: false,
    characterCollision: false,
  });
  expect(collisionWorld.objects.find((item) => item.id === "sam3_pillow_01")).toMatchObject({
    ready: false,
    mode: "none",
    faces: 0,
    bounds: null,
  });
  expect(await page.evaluate(() => window.__prepareRobotCollisionTest("sam3_plant_02")))
    .toMatchObject({ prepared: false, reason: "character_collision_not_approved" });

  const interpenetrationReport = await page.evaluate(() => window.__inspectSceneInterpenetrations());
  expect(interpenetrationReport).toMatchObject({
    targetCount: 3,
    intersectionCount: expect.any(Number),
    intersections: expect.any(Array),
  });
  expect(["passed", "failed"]).toContain(interpenetrationReport.status);
  expect(interpenetrationReport.meshPairTests).toBeGreaterThan(0);
  for (const intersection of interpenetrationReport.intersections) {
    expect(intersection.colliderIds).toHaveLength(2);
    expect(intersection.colliderIds[0]).not.toBe(intersection.colliderIds[1]);
  }

  const glbObject = await page.evaluate(() => window.__inspectInteractiveObject("sam3_plant_01"));
  const glbRayHit = await page.evaluate(({ center }) => {
    const origin = window.__visualPhysicsDemoState.cameraPosition;
    const direction = [center.x - origin.x, center.y - origin.y, center.z - origin.z];
    return window.__probeCollisionRay([origin.x, origin.y, origin.z], direction, 100);
  }, { center: glbObject.collisionBounds.center });
  expect(glbRayHit).toMatchObject({
    colliderKind: "object",
    objectId: "sam3_plant_01",
  });

  await page.evaluate(() => window.__focusSceneEntity("sam3_plant_01"));
  await page.waitForTimeout(900);
  const plantPointerTarget = await page.evaluate(
    () => window.__getInteractiveObjectScreenPoint("sam3_plant_01"),
  );
  const plantBeforeDrag = await page.evaluate(() => ({
    object: window.__inspectInteractiveObject("sam3_plant_01"),
    interaction: window.__visualPhysicsDemoState.interactiveObjects
      .find((item) => item.id === "sam3_plant_01"),
  }));
  expect(plantBeforeDrag.interaction).toMatchObject({
    interactionKind: "spin",
    interactionDrag: null,
  });
  await page.mouse.move(plantPointerTarget.x, plantPointerTarget.y);
  await page.mouse.down();
  await page.mouse.move(plantPointerTarget.x + 72, plantPointerTarget.y, { steps: 6 });
  await page.mouse.up();
  const plantAfterDrag = await page.evaluate(() => ({
    object: window.__inspectInteractiveObject("sam3_plant_01"),
    activeDrag: window.__visualPhysicsDemoState.activeObjectDrag,
  }));
  expect(plantAfterDrag.activeDrag).toBeNull();
  expect(plantAfterDrag.object.groupMatrixWorld).toEqual(plantBeforeDrag.object.groupMatrixWorld);
  expect(plantAfterDrag.object.splatMatrixWorld).toEqual(plantBeforeDrag.object.splatMatrixWorld);
  expect(plantAfterDrag.object.proxyMatrixWorld).toEqual(plantBeforeDrag.object.proxyMatrixWorld);

  const pillowBefore = await page.evaluate(() => window.__inspectInteractiveObject("sam3_pillow_01"));
  expect(pillowBefore).toMatchObject({
    visualReady: true,
    colliderReady: false,
    visualKind: "rgb-points",
    collisionMode: "none",
    collisionFaces: 0,
    collisionBounds: null,
  });
  expect(pillowBefore.splatWorldBounds).not.toBeNull();
  expect(pillowBefore.proxyWorldBounds).not.toBeNull();

  await page.evaluate(() => window.__focusSceneEntity("sam3_pillow_01"));
  await page.waitForTimeout(1_500);
  const pointerTarget = await page.evaluate(() => window.__getInteractiveObjectScreenPoint("sam3_pillow_01"));
  expect(pointerTarget).toBeTruthy();
  const beforeDrag = await page.evaluate(() => ({
    object: window.__inspectInteractiveObject("sam3_pillow_01"),
    camera: window.__visualPhysicsDemoState.cameraPosition,
    target: window.__visualPhysicsDemoState.cameraTarget,
  }));
  await page.mouse.move(pointerTarget.x, pointerTarget.y);
  await page.mouse.down();
  await page.mouse.move(pointerTarget.x + 72, pointerTarget.y, { steps: 6 });
  await page.mouse.up();
  const afterDrag = await page.evaluate(() => ({
    object: window.__inspectInteractiveObject("sam3_pillow_01"),
    camera: window.__visualPhysicsDemoState.cameraPosition,
    target: window.__visualPhysicsDemoState.cameraTarget,
    activeDrag: window.__visualPhysicsDemoState.activeObjectDrag,
  }));
  expect(afterDrag.activeDrag).toBeNull();
  expect(afterDrag.object.groupMatrixWorld).not.toEqual(beforeDrag.object.groupMatrixWorld);
  expect(afterDrag.object.splatMatrixWorld).not.toEqual(beforeDrag.object.splatMatrixWorld);
  expect(afterDrag.object.proxyMatrixWorld).not.toEqual(beforeDrag.object.proxyMatrixWorld);
  expectVectorClose(afterDrag.camera, beforeDrag.camera);
  expectVectorClose(afterDrag.target, beforeDrag.target);

  const preSpinMatrix = afterDrag.object.groupMatrixWorld;
  const pointerTargetAfterDrag = await page.evaluate(() => window.__getInteractiveObjectScreenPoint("sam3_pillow_01"));
  await page.mouse.dblclick(pointerTargetAfterDrag.x, pointerTargetAfterDrag.y, { delay: 45 });
  await page.waitForFunction(() => window.__visualPhysicsDemoState.interactiveObjectTurns >= 1);
  const afterSpin = await page.evaluate(() => window.__inspectInteractiveObject("sam3_pillow_01"));
  expect(afterSpin.groupMatrixWorld).toEqual(preSpinMatrix);

  const yawResult = await page.evaluate(() => window.__setInteractiveObjectYaw("sam3_pillow_01", 90));
  expect(yawResult.updated).toBe(true);
  const afterYaw = await page.evaluate(() => window.__inspectInteractiveObject("sam3_pillow_01"));
  expect(afterYaw.groupMatrixWorld).not.toEqual(afterSpin.groupMatrixWorld);

  expect(consoleProblems).toEqual([]);
});

test("scene mutations submit a structured queue request only when the manifest configures a backend", async ({ page }) => {
  const serviceManifest = runtimeFixture();
  serviceManifest.sceneCommandService = {
    contract: "video2world-scene-command-service-1.0.0",
    endpoint: "/v1/scene-commands/submit",
  };
  let submitted = null;
  await page.route((url) => url.pathname === "/test-fixtures/manifest.json", async (route) => {
    await route.fulfill({ json: serviceManifest });
  });
  await page.route((url) => url.pathname === "/v1/scene-commands/plan", async (route) => {
    const planned = route.request().postDataJSON();
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        status: "ready",
        manifestSha256: "a".repeat(64),
        confirmationRequired: true,
        confirmationPhrase: `confirm ${planned.command.request_id}`,
        plan: {
          status: "ready",
          affected_stages: planned.clientPreview.affectedStages.map((stage) => ({ stage })),
        },
      }),
    });
  });
  await page.route((url) => url.pathname === "/v1/scene-commands/submit", async (route) => {
    submitted = route.request().postDataJSON();
    await route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify({
        status: "queued",
        requestId: submitted.command.request_id,
        planId: "sceneplan_0123456789abcdef0123",
        idempotencyKey: "c".repeat(64),
        manifestSha256: "d".repeat(64),
        jobId: "scenejob_fixture",
      }),
    });
  });

  await page.goto(fixtureUrl, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__visualPhysicsDemoState?.sceneQaReady);
  const objectCountBefore = await page.evaluate(() => window.__visualPhysicsDemoState.interactiveObjectCount);
  await page.locator("#sceneQaInput").fill("删除左侧盆栽");
  await page.locator("#sceneQaSubmit").click();
  await expect(page.locator("#sceneCommandConfirm")).toBeEnabled();
  await expect(page.locator("#sceneCommandStatus")).toContainText("后端计划已验证");
  await expect(page.locator("#sceneCommandStages span")).toHaveCount(6);
  await page.locator("#sceneCommandConfirm").click();
  await expect(page.locator("#sceneCommandStatus")).toContainText("已进入 pipeline 队列");

  expect(submitted).toMatchObject({
    rawPrompt: "删除左侧盆栽",
    structuredIntent: {
      kind: "delete_object",
      target: { object_id: "sam3_plant_01" },
    },
    clientPreview: {
      status: "ready",
      previewOnly: true,
      manifestVersion: "browser-fixture-v1",
      targetIds: ["sam3_plant_01"],
    },
  });
  expect(submitted.command.intent).toEqual(submitted.structuredIntent);
  expect(submitted.command.request_id).toMatch(/^web-/);
  expect(submitted.confirmationPhrase).toBe(`confirm ${submitted.command.request_id}`);
  expect(await page.evaluate(() => window.__visualPhysicsDemoState.interactiveObjectCount))
    .toBe(objectCountBefore);
  expect(await page.evaluate(() => window.__visualPhysicsDemoState.sceneCommandStatus))
    .toBe("accepted");
});

test("selection-only objects can focus but cannot drag or spin", async ({ page }) => {
  const manifest = runtimeFixture();
  delete manifest.interactiveObjects[0].interaction;
  await page.route(
    (url) => url.pathname === "/test-fixtures/selection-only-manifest.json",
    (route) => route.fulfill({ json: manifest }),
  );

  await page.goto("/?visual=off&manifest=/test-fixtures/selection-only-manifest.json", {
    waitUntil: "domcontentloaded",
  });
  await page.waitForFunction(() => typeof window.__loadInteractiveObjectVisuals === "function");
  await page.evaluate(() => window.__loadInteractiveObjectVisuals());
  await page.waitForFunction(() => window.__visualPhysicsDemoState?.interactiveObjectReadyCount === 3);
  const focused = await page.evaluate(() => window.__focusSceneEntity("sam3_plant_01"));
  expect(focused.focused).toBe(true);
  await page.waitForTimeout(900);

  const point = await page.evaluate(() => window.__getInteractiveObjectScreenPoint("sam3_plant_01"));
  const before = await page.evaluate(() => window.__inspectInteractiveObject("sam3_plant_01"));
  expect(before).toMatchObject({ interactionKind: null, interactionDrag: null });
  expect(await page.evaluate(() => window.__spinInteractiveObject("sam3_plant_01"))).toMatchObject({
    started: false,
  });

  await page.mouse.move(point.x, point.y);
  await page.mouse.down();
  await page.mouse.move(point.x + 64, point.y, { steps: 5 });
  await page.mouse.up();
  const afterDrag = await page.evaluate(() => window.__inspectInteractiveObject("sam3_plant_01"));
  expect(afterDrag.groupMatrixWorld).toEqual(before.groupMatrixWorld);

  const pointAfterOrbit = await page.evaluate(
    () => window.__getInteractiveObjectScreenPoint("sam3_plant_01"),
  );
  await page.mouse.dblclick(pointAfterOrbit.x, pointAfterOrbit.y, { delay: 45 });
  await page.waitForTimeout(550);
  const afterDoubleClick = await page.evaluate(
    () => window.__inspectInteractiveObject("sam3_plant_01"),
  );
  expect(afterDoubleClick.turns).toBe(0);
  expect(afterDoubleClick.groupMatrixWorld).toEqual(before.groupMatrixWorld);
});

test("candidate GLB review never grants character collision without a passed gate", async ({ page }) => {
  const manifest = runtimeFixture();
  Object.assign(manifest.interactiveObjects[0].collision.gate, {
    status: "failed",
    reason: "fixture gate intentionally failed",
  });
  await page.route(
    (url) => url.pathname === "/test-fixtures/candidate-gate-manifest.json",
    (route) => route.fulfill({ json: manifest }),
  );

  await page.goto(
    "/?visual=off&allowCandidateColliders=1&manifest=/test-fixtures/candidate-gate-manifest.json",
    { waitUntil: "domcontentloaded" },
  );
  await page.waitForFunction(() => {
    const state = window.__visualPhysicsDemoState;
    return state?.colliderReady && state?.objectColliderReadyCount === 2;
  });

  const collisionWorld = await page.evaluate(() => window.__inspectCollisionWorld());
  expect(collisionWorld.objects.find((item) => item.id === "sam3_plant_01")).toMatchObject({
    mode: "glb",
    gateStatus: "failed",
    walkable: false,
    characterCollision: false,
  });
  expect(await page.evaluate(() => window.__prepareRobotCollisionTest("sam3_plant_01")))
    .toMatchObject({ prepared: false, reason: "character_collision_not_approved" });
});

test("one surface-BVH GLB is the PBR visual, selection target, and character collider", async ({ page }) => {
  const manifest = runtimeFixture();
  makeUnifiedObject(manifest.interactiveObjects[0]);
  let unifiedAssetRequests = 0;
  page.on("request", (request) => {
    if (new URL(request.url()).pathname.endsWith("/test-fixtures/tiny-collider.glb.base64.txt")) {
      unifiedAssetRequests += 1;
    }
  });
  await page.route(
    (url) => url.pathname === "/test-fixtures/mesh-first-manifest.json",
    (route) => route.fulfill({ json: manifest }),
  );

  await page.goto("/?visual=off&manifest=/test-fixtures/mesh-first-manifest.json", {
    waitUntil: "domcontentloaded",
  });
  await page.waitForFunction(() => typeof window.__loadInteractiveObjectVisuals === "function");
  const loaded = await page.evaluate(() => window.__loadInteractiveObjectVisuals());
  expect(loaded).toMatchObject({
    interactiveObjectReadyCount: 3,
    interactiveObjectGaussianCount: 0,
    interactiveObjectRgbPointCount: 16,
    interactiveObjectMeshVertexCount: 3,
    interactiveObjectVisualPrimitiveCount: 19,
  });
  expect(unifiedAssetRequests).toBe(1);

  const focused = await page.evaluate(() => window.__focusSceneEntity("sam3_plant_01"));
  expect(focused.focused).toBe(true);
  await page.waitForTimeout(900);
  const before = await page.evaluate(() => window.__inspectInteractiveObject("sam3_plant_01"));
  expect(before).toMatchObject({
    visualReady: true,
    visualKind: "mesh",
    meshVertexCount: 3,
    interactionKind: "spin",
    interactionDrag: null,
    collisionMode: "unified-glb",
    collisionTopology: "surface_bvh",
    unifiedVisualCollision: true,
    proxyMatrixWorld: null,
    characterCollision: true,
    bvhMeshCount: 1,
    pbrMaterialCount: 1,
  });
  expect(before.materialTypes).toContain("MeshStandardMaterial");
  expect(before.splatLocalBounds).not.toBeNull();
  expect(before.splatWorldBounds).not.toBeNull();
  expect(before.collisionMatrixWorld).not.toBeNull();
  const collisionWorld = await page.evaluate(() => window.__inspectCollisionWorld());
  expect(collisionWorld.objects.find((item) => item.id === "sam3_plant_01")).toMatchObject({
    mode: "unified-glb",
    gateStatus: "passed",
    topology: "surface_bvh",
    volumePhysics: false,
    characterCollision: true,
    faces: 1,
  });

  const pointerTarget = await page.evaluate(
    () => window.__getInteractiveObjectScreenPoint("sam3_plant_01"),
  );
  expect(await page.evaluate(
    ({ x, y }) => window.__inspectInteractiveObjectHitAt(x, y),
    pointerTarget,
  )).toEqual({
    objectId: "sam3_plant_01",
    colliderMode: "unified-glb",
    collisionTopology: "surface_bvh",
  });
  await page.mouse.dblclick(pointerTarget.x, pointerTarget.y, { delay: 45 });
  await page.waitForFunction(() => window.__visualPhysicsDemoState.interactiveObjects
    .find((item) => item.id === "sam3_plant_01")?.spinning === true);
  await page.waitForTimeout(120);
  const duringSpin = await page.evaluate(() => window.__inspectInteractiveObject("sam3_plant_01"));
  expect(duringSpin.splatMatrixWorld).not.toEqual(before.splatMatrixWorld);
  expect(duringSpin.collisionMatrixWorld).not.toEqual(before.collisionMatrixWorld);
});

test("an unbaked unified GLB keeps nonuniform placement below a rigid interactive root", async ({ page }) => {
  const manifest = runtimeFixture();
  const object = makeUnifiedObject(manifest.interactiveObjects[0]);
  const [x, y, z] = object.placement.pivot;
  Object.assign(object.placement, {
    generatedCenter: [0, 0, 0],
    scale: [1, 1, 1],
    rotationEulerDeg: [0, 0, 0],
    matrixRowMajor: [
      [0, -3, 0, x],
      [2, 0, 0, y],
      [0, 0, 4, z],
      [0, 0, 0, 1],
    ],
  });
  await page.route(
    (url) => url.pathname === "/test-fixtures/unified-matrix-manifest.json",
    (route) => route.fulfill({ json: manifest }),
  );
  await page.goto("/?visual=off&manifest=/test-fixtures/unified-matrix-manifest.json", {
    waitUntil: "domcontentloaded",
  });
  await page.waitForFunction(() => typeof window.__loadInteractiveObjectVisuals === "function");
  await page.evaluate(() => window.__loadInteractiveObjectVisuals());
  const before = await page.evaluate(() => window.__inspectInteractiveObject("sam3_plant_01"));
  expect(before.groupMatrix).toEqual([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
  ]);
  expect(matrixGram(before.splatMatrixWorld)).toEqual([4, 9, 16, 0, 0, 0]);
  expect(before).toMatchObject({
    unifiedVisualCollision: true,
    collisionMode: "unified-glb",
    collisionTopology: "surface_bvh",
  });

  const yaw = await page.evaluate(() => window.__setInteractiveObjectYaw("sam3_plant_01", 90));
  expect(yaw.updated).toBe(true);
  const after = await page.evaluate(() => window.__inspectInteractiveObject("sam3_plant_01"));
  expect(after.groupMatrix).not.toEqual(before.groupMatrix);
  expect(after.groupMatrix.slice(12, 15)).toEqual([x, y, z]);
  expect(matrixGram(after.groupMatrix)).toEqual([1, 1, 1, 0, 0, 0]);
  expect(matrixGram(after.splatMatrixWorld)).toEqual(matrixGram(before.splatMatrixWorld));
  expect(after.splatMatrixWorld).not.toEqual(before.splatMatrixWorld);
  expect(after.collisionMatrixWorld).not.toEqual(before.collisionMatrixWorld);
});

test("unified GLB geometry mismatch fails closed without a box fallback", async ({ page }) => {
  const manifest = runtimeFixture();
  const object = makeUnifiedObject(manifest.interactiveObjects[0]);
  object.collision.asset.faces = 2;
  await page.route(
    (url) => url.pathname === "/test-fixtures/unified-mismatch-manifest.json",
    (route) => route.fulfill({ json: manifest }),
  );

  await page.goto("/?visual=off&manifest=/test-fixtures/unified-mismatch-manifest.json", {
    waitUntil: "domcontentloaded",
  });
  await page.waitForFunction(() => window.__visualPhysicsDemoState?.error?.includes(
    "unified-glb face count 1 does not match manifest 2",
  ));
  const collisionWorld = await page.evaluate(() => window.__inspectCollisionWorld());
  expect(collisionWorld).toMatchObject({
    objectColliderReadyCount: 0,
    degradedCount: 0,
  });
  expect(collisionWorld.objects.find((item) => item.id === "sam3_plant_01")).toMatchObject({
    ready: false,
    mode: "none",
    characterCollision: false,
    bounds: null,
  });
  expect(await page.evaluate(() => window.__inspectInteractiveObject("sam3_plant_01")))
    .toMatchObject({
      visualReady: false,
      colliderReady: false,
      collisionMode: "none",
      proxyMatrixWorld: null,
    });
});

test("a failed unified surface-collision gate is rejected before any mesh or proxy loads", async ({ page }) => {
  const manifest = runtimeFixture();
  const object = makeUnifiedObject(manifest.interactiveObjects[0]);
  object.collision.gate.status = "failed";
  await page.route(
    (url) => url.pathname === "/test-fixtures/unified-failed-gate-manifest.json",
    (route) => route.fulfill({ json: manifest }),
  );

  await page.goto("/?visual=off&manifest=/test-fixtures/unified-failed-gate-manifest.json", {
    waitUntil: "domcontentloaded",
  });
  await page.waitForFunction(() => window.__visualPhysicsDemoState?.error?.includes(
    "unified-glb requires passed gate.status and gate.surfaceCollision",
  ));
  expect(await page.evaluate(() => window.__inspectCollisionWorld())).toMatchObject({
    sceneReady: false,
    objectColliderReadyCount: 0,
    degradedCount: 0,
    objects: [],
  });
});

test("parent rotation carries a child while child rotation leaves the parent unchanged", async ({ page }) => {
  const consoleProblems = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) consoleProblems.push(`${message.type()}: ${message.text()}`);
  });
  page.on("pageerror", (error) => consoleProblems.push(`pageerror: ${error.message}`));
  await page.route((url) => url.pathname === "/test-fixtures/hierarchy-manifest.json", (route) => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify(hierarchyFixture()),
  }));

  await page.goto("/?visual=off&manifest=/test-fixtures/hierarchy-manifest.json", {
    waitUntil: "domcontentloaded",
  });
  await page.waitForFunction(() => typeof window.__loadInteractiveObjectVisuals === "function");
  await page.evaluate(() => window.__loadInteractiveObjectVisuals());
  await page.waitForFunction(() => window.__visualPhysicsDemoState?.interactiveObjectReadyCount === 3);

  const before = await page.evaluate(() => ({
    parent: window.__inspectInteractiveObject("sam3_plant_01"),
    child: window.__inspectInteractiveObject("sam3_pillow_01"),
    childHierarchy: window.__visualPhysicsDemoState.interactiveObjects
      .find((item) => item.id === "sam3_pillow_01"),
    parentHierarchy: window.__visualPhysicsDemoState.interactiveObjects
      .find((item) => item.id === "sam3_plant_01"),
    transform: {
      scale: window.__visualPhysicsDemoState.transformScale,
      rotation: window.__visualPhysicsDemoState.transformRotationMatrix,
      translation: window.__visualPhysicsDemoState.transformTranslation,
    },
  }));
  expect(before.childHierarchy).toMatchObject({
    semanticGranularity: "independent_child_asset",
    parentObjectId: "sam3_plant_01",
    movesWithParent: true,
    independentlyMovable: true,
  });
  expect(before.parentHierarchy.childObjectIds).toContain("sam3_pillow_01");
  expect(before.child.hierarchyAttachmentMatrixDelta).toBeLessThan(0.00001);
  expect(matrixGram(before.child.groupMatrixWorld)).toEqual([1, 1, 1, 0, 0, 0]);
  const sourcePivot = [0, 0, 15.3];
  const transformedPivot = ["x", "y", "z"].reduce((result, axis, row) => ({
    ...result,
    [axis]: before.transform.translation[axis] + before.transform.scale
      * before.transform.rotation[row].reduce(
        (sum, coefficient, column) => sum + coefficient * sourcePivot[column],
        0,
      ),
  }), {});
  expectVectorClose(before.childHierarchy.position, transformedPivot);

  const parentYaw = await page.evaluate(() => window.__setInteractiveObjectYaw("sam3_plant_01", 90));
  expect(parentYaw.updated).toBe(true);
  const afterParentYaw = await page.evaluate(() => ({
    parent: window.__inspectInteractiveObject("sam3_plant_01"),
    child: window.__inspectInteractiveObject("sam3_pillow_01"),
  }));
  expect(afterParentYaw.parent.groupMatrixWorld).not.toEqual(before.parent.groupMatrixWorld);
  expect(afterParentYaw.child.groupMatrixWorld).not.toEqual(before.child.groupMatrixWorld);
  expect(afterParentYaw.child.splatMatrixWorld).not.toEqual(before.child.splatMatrixWorld);
  expect(matrixGram(afterParentYaw.child.splatMatrixWorld)).toEqual(
    matrixGram(before.child.splatMatrixWorld),
  );

  await page.evaluate(() => window.__setInteractiveObjectYaw("sam3_plant_01", 0));
  const beforeChildYaw = await page.evaluate(() => ({
    parent: window.__inspectInteractiveObject("sam3_plant_01"),
    child: window.__inspectInteractiveObject("sam3_pillow_01"),
  }));
  const childYaw = await page.evaluate(() => window.__setInteractiveObjectYaw("sam3_pillow_01", 90));
  expect(childYaw.updated).toBe(true);
  const afterChildYaw = await page.evaluate(() => ({
    parent: window.__inspectInteractiveObject("sam3_plant_01"),
    child: window.__inspectInteractiveObject("sam3_pillow_01"),
  }));
  expect(afterChildYaw.parent.groupMatrixWorld).toEqual(beforeChildYaw.parent.groupMatrixWorld);
  expect(afterChildYaw.child.groupMatrixWorld).not.toEqual(beforeChildYaw.child.groupMatrixWorld);
  expect(matrixGram(afterChildYaw.child.splatMatrixWorld)).toEqual(
    matrixGram(beforeChildYaw.child.splatMatrixWorld),
  );
  expect(consoleProblems).toEqual([]);
});

test("logical hierarchy ancestors do not load colliders or accept focus", async ({ page }) => {
  const manifest = logicalHierarchyFixture();
  await page.route(
    (url) => url.pathname === "/test-fixtures/logical-hierarchy-manifest.json",
    (route) => route.fulfill({ json: manifest }),
  );
  await page.goto("/?visual=off&manifest=/test-fixtures/logical-hierarchy-manifest.json", {
    waitUntil: "domcontentloaded",
  });
  await page.waitForFunction(() => typeof window.__loadInteractiveObjectVisuals === "function");
  await page.evaluate(() => window.__loadInteractiveObjectVisuals());
  await page.waitForFunction(() => window.__visualPhysicsDemoState?.interactiveObjects?.length === 3);

  const before = await page.evaluate(() => ({
    state: window.__visualPhysicsDemoState,
    ancestor: window.__inspectInteractiveObject("sam3_plant_01"),
    child: window.__inspectInteractiveObject("sam3_pillow_01"),
  }));
  expect(before.state.interactiveObjectCount).toBe(2);
  expect(before.ancestor).toMatchObject({
    logicalHierarchyOnly: true,
    visualReady: false,
    colliderReady: false,
    collisionMode: "none",
    childObjectIds: ["sam3_pillow_01"],
  });
  expect(before.child).toMatchObject({
    parentObjectId: "sam3_plant_01",
    movesWithParent: true,
    independentlyMovable: true,
  });

  const ancestorFocus = await page.evaluate(() => window.__focusSceneEntity("sam3_plant_01"));
  expect(ancestorFocus.focused).toBe(false);
  expect(ancestorFocus.state.selectedSceneEntity).toBeNull();
  expect(ancestorFocus.state.selectedInteractiveObject).toBeNull();

  const childFocus = await page.evaluate(() => window.__focusSceneEntity("sam3_pillow_01"));
  expect(childFocus.focused).toBe(true);
  expect(childFocus.state.selectedSceneEntity).toBe("sam3_pillow_01");
  expect(childFocus.state.selectedInteractiveObject).toBe("sam3_pillow_01");
});

test("camera presets frame a bed-sized overview target outside every interactive AABB", async ({ page }) => {
  const manifest = cameraOverviewFixture();
  await page.route(
    (url) => url.pathname === "/test-fixtures/camera-overview-manifest.json",
    (route) => route.fulfill({ json: manifest }),
  );
  await page.goto("/?visual=off&manifest=/test-fixtures/camera-overview-manifest.json", {
    waitUntil: "domcontentloaded",
  });
  await page.waitForFunction(() => {
    const state = window.__visualPhysicsDemoState;
    return state?.objectColliderCount > 0
      && state.objectColliderReadyCount === state.objectColliderCount;
  });

  const presets = ["reference", "doorway", "front", "back", "left", "right", "top"];
  const views = {};
  for (const preset of presets) {
    views[preset] = await page.evaluate((presetId) => {
      const first = window.__setCameraPreset(presetId);
      const repeated = window.__setCameraPreset(presetId);
      const target = window.__inspectInteractiveObject("sam3_plant_01");
      return {
        first: {
          position: first.cameraPosition,
          target: first.cameraTarget,
          coverage: first.cameraOverviewCoverage,
        },
        repeated: {
          position: repeated.cameraPosition,
          target: repeated.cameraTarget,
          coverage: repeated.cameraOverviewCoverage,
        },
        focusObjectId: first.cameraOverviewFocusObjectId,
        insideObjectIds: first.cameraInsideInteractiveObjectIds,
        presetSource: first.cameraPresetSource,
        targetBoundsCenter: target.collisionBounds.center,
      };
    }, preset);

    expect(views[preset].focusObjectId).toBe("sam3_plant_01");
    expect(views[preset].insideObjectIds).toEqual([]);
    expect(views[preset].presetSource).toContain("+overview:sam3_plant_01");
    expect(views[preset].first).toEqual(views[preset].repeated);
    expectVectorClose(views[preset].first.target, views[preset].targetBoundsCenter, 0.0002);
    expect(views[preset].first.coverage.widthFraction).toBeGreaterThan(0.05);
    expect(views[preset].first.coverage.widthFraction).toBeLessThan(0.92);
    expect(views[preset].first.coverage.heightFraction).toBeGreaterThan(0.05);
    expect(views[preset].first.coverage.heightFraction).toBeLessThan(0.92);
    expect(views[preset].first.coverage.minimumNdcZ).toBeGreaterThanOrEqual(-1);
    expect(views[preset].first.coverage.maximumNdcZ).toBeLessThanOrEqual(1);
  }

  const horizontalDirection = (view) => {
    const { position, target } = view.first;
    const x = position.x - target.x;
    const z = position.z - target.z;
    const length = Math.hypot(x, z);
    return { x: x / length, z: z / length };
  };
  const dot = (left, right) => left.x * right.x + left.z * right.z;
  expect(dot(horizontalDirection(views.front), horizontalDirection(views.back))).toBeLessThan(-0.8);
  expect(dot(horizontalDirection(views.left), horizontalDirection(views.right))).toBeLessThan(-0.8);
  const topOffset = {
    x: views.top.first.position.x - views.top.first.target.x,
    y: views.top.first.position.y - views.top.first.target.y,
    z: views.top.first.position.z - views.top.first.target.z,
  };
  expect(Math.abs(topOffset.y) / Math.hypot(topOffset.x, topOffset.y, topOffset.z)).toBeGreaterThan(0.82);
});
