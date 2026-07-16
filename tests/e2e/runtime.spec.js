import { expect, test } from "@playwright/test";

const fixtureUrl = "/?visual=off&manifest=/test-fixtures/manifest.json";

function expectVectorClose(actual, expected, epsilon = 0.0002) {
  for (const key of ["x", "y", "z"]) {
    expect(Math.abs(actual[key] - expected[key])).toBeLessThan(epsilon);
  }
}

test("fresh-clone root loads the committed fixture without local production assets", async ({ page }) => {
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
    faces: 1,
  });
  expect(collisionWorld.objects.find((item) => item.id === "sam3_plant_02")).toMatchObject({
    mode: "degraded-box",
  });
  expect(collisionWorld.objects.find((item) => item.id === "sam3_pillow_01")).toMatchObject({
    ready: false,
    mode: "none",
    faces: 0,
    bounds: null,
  });

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
  await page.waitForTimeout(900);
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
