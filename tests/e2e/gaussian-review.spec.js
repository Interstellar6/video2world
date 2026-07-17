import { expect, test } from "@playwright/test";

const fixtureUrl = "/gaussian-review.html?asset=/test-fixtures/tiny-standard-gaussian.ply"
  + "&objectId=tiny_standard_gaussian&expectedColor=light-neutral"
  + "&expectedExtents=1.8,1,0.5&extentTolerance=0.02&minThicknessRatio=0.2";

test("standard GraphDECO Gaussian PLY renders deterministic nonblank six-view evidence", async ({ page }) => {
  const consoleProblems = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) consoleProblems.push(`${message.type()}: ${message.text()}`);
  });
  page.on("pageerror", (error) => consoleProblems.push(`pageerror: ${error.message}`));

  await page.goto(fixtureUrl, { waitUntil: "domcontentloaded" });
  await expect(page).toHaveTitle("Video2World Gaussian six-view review");
  await page.waitForFunction(() => ["ready", "error"].includes(window.__VIDEO2WORLD_GAUSSIAN_REVIEW__?.state));
  const review = await page.evaluate(() => {
    const state = structuredClone(window.__VIDEO2WORLD_GAUSSIAN_REVIEW__);
    delete state.renderDataUrls;
    return state;
  });
  expect(review.error).toBeNull();
  expect(review.renderMode).toBe("spark-sh0-direct-color");
  expect(review.pointCount).toBe(90);
  expect(review.sourceBounds.size[0]).toBeCloseTo(1.8, 3);
  expect(review.sourceBounds.size[1]).toBeCloseTo(1, 3);
  expect(review.sourceBounds.size[2]).toBeCloseTo(0.5, 3);
  expect(review.gates).toEqual({
    allSixViewsRendered: true,
    allSixViewsNonblank: true,
    directColorMatchesExpectation: true,
    notSheetlike: true,
    boundsMatchExpectation: true,
    boundsFrameAllViews: true,
  });
  expect(review.promotionAllowed).toBe(true);
  expect(review.blockers).toEqual([]);
  expect(Object.values(review.viewMetrics)).toHaveLength(6);
  for (const metrics of Object.values(review.viewMetrics)) {
    expect(metrics.nonblank).toBe(true);
    expect(metrics.nonTransparentPixels).toBeGreaterThan(64);
    expect(metrics.meanLuminance).toBeGreaterThan(155);
  }
  await expect(page.locator("[data-review-render]")).toHaveCount(6);
  const renderSources = await page.locator("[data-review-render]").evaluateAll((images) => (
    images.map((image) => image.getAttribute("src"))
  ));
  expect(renderSources.every((source) => source?.startsWith("data:image/png;base64,"))).toBe(true);
  const canvasPixels = await page.locator("[data-review-canvas]").evaluateAll((canvases) => canvases.map((canvas) => {
    const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let nonblank = 0;
    for (let index = 3; index < pixels.length; index += 4) nonblank += Number(pixels[index] > 8);
    return nonblank;
  }));
  expect(canvasPixels).toHaveLength(6);
  expect(canvasPixels.every((count) => count > 64)).toBe(true);
  await expect(page.locator("#review-status")).toContainText("passed all configured gates");
  expect(consoleProblems).toEqual([]);
});

test("blank framing, wrong color, sheetlike geometry, and wrong bounds block promotion", async ({ page }) => {
  const rejectionUrl = "/gaussian-review.html?asset=/test-fixtures/tiny-standard-gaussian.ply"
    + "&objectId=tiny_standard_gaussian&expectedColor=light-neutral&center=50,50,50"
    + "&expectedMeanRgb=20,20,20&maxMeanRgbError=1"
    + "&expectedExtents=3,3,3&extentTolerance=0.01&minThicknessRatio=0.4";
  await page.goto(rejectionUrl, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => ["ready", "error"].includes(window.__VIDEO2WORLD_GAUSSIAN_REVIEW__?.state));
  const review = await page.evaluate(() => {
    const state = structuredClone(window.__VIDEO2WORLD_GAUSSIAN_REVIEW__);
    delete state.renderDataUrls;
    return state;
  });
  expect(review.error).toBeNull();
  expect(review.promotionAllowed).toBe(false);
  expect(review.gates).toMatchObject({
    allSixViewsRendered: true,
    allSixViewsNonblank: false,
    directColorMatchesExpectation: false,
    notSheetlike: false,
    boundsMatchExpectation: false,
    boundsFrameAllViews: false,
  });
  expect(review.blockers).toEqual(expect.arrayContaining([
    "allSixViewsNonblank",
    "directColorMatchesExpectation",
    "notSheetlike",
    "boundsMatchExpectation",
    "boundsFrameAllViews",
  ]));
  await expect(page.locator("#review-status")).toContainText("promotion blocked");
});
