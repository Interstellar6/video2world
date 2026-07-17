import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { chromium } from "@playwright/test";

function argument(name, fallback = null) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

const url = argument("url");
const outputDir = path.resolve(argument("output-dir", "."));
const screenshotName = argument("screenshot", "gaussian_six_view_contact_sheet.png");
const receiptName = argument("receipt", "gaussian_six_view_review.json");
if (!url) throw new Error("Usage: node scripts/capture_gaussian_review.mjs --url <review-url> --output-dir <dir>");

const browser = await chromium.launch({ channel: "chrome", headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1280 }, deviceScaleFactor: 1 });
  const consoleProblems = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) consoleProblems.push(`${message.type()}: ${message.text()}`);
  });
  page.on("pageerror", (error) => consoleProblems.push(`pageerror: ${error.message}`));
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => ["ready", "error"].includes(window.__VIDEO2WORLD_GAUSSIAN_REVIEW__?.state), null, {
    timeout: 60_000,
  });
  const state = await page.evaluate(() => {
    const value = structuredClone(window.__VIDEO2WORLD_GAUSSIAN_REVIEW__);
    delete value.renderDataUrls;
    return value;
  });
  if (state.state !== "ready") throw new Error(`Gaussian review failed: ${state.error}`);
  fs.mkdirSync(outputDir, { recursive: true });
  const screenshotPath = path.join(outputDir, screenshotName);
  await page.locator("#review-grid").screenshot({ path: screenshotPath });
  const screenshotBytes = fs.readFileSync(screenshotPath);
  const receipt = {
    ...state,
    createdAt: new Date().toISOString(),
    browserEvidence: {
      contactSheet: screenshotName,
      bytes: screenshotBytes.byteLength,
      sha256: crypto.createHash("sha256").update(screenshotBytes).digest("hex"),
      viewport: [1440, 1280],
      consoleProblems,
    },
  };
  fs.writeFileSync(path.join(outputDir, receiptName), `${JSON.stringify(receipt, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({
    state: receipt.state,
    promotionAllowed: receipt.promotionAllowed,
    blockers: receipt.blockers,
    screenshot: screenshotPath,
    receipt: path.join(outputDir, receiptName),
  }, null, 2)}\n`);
} finally {
  await browser.close();
}
