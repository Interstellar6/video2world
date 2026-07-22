import { describe, expect, it } from "vitest";

import {
  assetFetchRetryDelayMs,
  buildAssetFetchAttemptPlan,
  DEFAULT_ASSET_FETCH_ATTEMPTS_PER_SOURCE,
} from "../../web/asset-fetch-policy.js";

describe("asset fetch retry policy", () => {
  it("retries each source before moving to the next one", () => {
    const origin = { key: "origin" };
    const mirror = { key: "mirror" };
    const plan = buildAssetFetchAttemptPlan([origin, mirror]);

    expect(DEFAULT_ASSET_FETCH_ATTEMPTS_PER_SOURCE).toBe(3);
    expect(plan).toHaveLength(6);
    expect(plan.map((attempt) => attempt.source.key)).toEqual([
      "origin", "origin", "origin", "mirror", "mirror", "mirror",
    ]);
    expect(plan.map((attempt) => attempt.cacheMode)).toEqual([
      "force-cache", "reload", "reload", "force-cache", "reload", "reload",
    ]);
    expect(plan.map((attempt) => attempt.attemptNumber)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(plan.every((attempt) => attempt.totalAttempts === 6)).toBe(true);
  });

  it("uses a bounded linear backoff within each source", () => {
    expect([0, 1, 2].map((index) => assetFetchRetryDelayMs(index, 650)))
      .toEqual([650, 1300, 1950]);
  });

  it("rejects empty sources and invalid retry counts", () => {
    expect(() => buildAssetFetchAttemptPlan([])).toThrow(/non-empty array/);
    expect(() => buildAssetFetchAttemptPlan([{ key: "origin" }], { attemptsPerSource: 0 }))
      .toThrow(/positive integer/);
  });
});
