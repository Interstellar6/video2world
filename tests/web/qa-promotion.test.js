import { describe, expect, it } from "vitest";
import { validatePromotionReport } from "../../scripts/qa_bedroom4_production_browser.mjs";

const manifest = {
  interactiveObjects: [
    { id: "plant", collision: { mode: "glb" } },
    { id: "pillow", collision: { mode: "none" } },
  ],
};

function report() {
  return {
    status: "candidate",
    automatedGate: "passed",
    manifestSha256: "current",
    failures: [],
    counts: { visualObjectsExpected: 2, objectCollidersExpected: 1 },
    objects: [{ id: "plant" }, { id: "pillow" }],
    collisionWorld: { objects: [{ id: "plant" }, { id: "pillow" }] },
    interactions: { robotCollisions: [{ objectId: "plant", passed: true }] },
  };
}

describe("production placement promotion", () => {
  it("accepts a current report with complete object and collider coverage", () => {
    expect(validatePromotionReport({
      browserReport: report(),
      manifest,
      currentManifestSha256: "current",
    })).toBe(true);
  });

  it("rejects a stale report hash", () => {
    expect(() => validatePromotionReport({
      browserReport: report(),
      manifest,
      currentManifestSha256: "changed",
    })).toThrow("current manifest hash");
  });

  it("rejects missing robot collision coverage", () => {
    const incomplete = report();
    incomplete.interactions.robotCollisions = [];
    expect(() => validatePromotionReport({
      browserReport: incomplete,
      manifest,
      currentManifestSha256: "current",
    })).toThrow("every collider");
  });
});
