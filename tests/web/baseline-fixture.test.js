import { describe, expect, it } from "vitest";
import {
  baselinePresentation,
  WEB_DEMO_BASELINE_252A85C,
} from "../../web/baseline-fixture.js";

describe("web demo baseline fixture", () => {
  it("pins the approved 252a85c camera and robot spawn", () => {
    expect(WEB_DEMO_BASELINE_252A85C.commit).toMatch(/^252a85c/);
    expect(WEB_DEMO_BASELINE_252A85C.camera.id).toBe("pgsr_input_camera_000018");
    expect(WEB_DEMO_BASELINE_252A85C.camera.verticalFovDeg).toBe(55.776917);
    expect(WEB_DEMO_BASELINE_252A85C.robotSpawn.groundPoint).toEqual([
      5.968559,
      6.051313,
      7.851893,
    ]);
  });

  it("uses the approved presentation when a manifest omits one", () => {
    expect(baselinePresentation({})).toEqual({
      pivot: [2.250915668599896, -3.375, 12.605660413500915],
      yawDeg: 180,
    });
  });

  it("accepts an explicit presentation for another world", () => {
    expect(baselinePresentation({ presentation: { pivot: [1, 2, 3], yawDeg: 90 } })).toEqual({
      pivot: [1, 2, 3],
      yawDeg: 90,
    });
  });
});
