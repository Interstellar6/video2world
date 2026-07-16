const freezeVector = (values) => Object.freeze([...values]);

export const WEB_DEMO_BASELINE_252A85C = Object.freeze({
  commit: "252a85c52106838b9bd796f4e70253f7e68af488",
  alignment: Object.freeze({
    scale: 1,
    rotationMatrix: Object.freeze([
      freezeVector([1, 0, 0]),
      freezeVector([0, 1, 0]),
      freezeVector([0, 0, 1]),
    ]),
    translation: freezeVector([0, 0, 0]),
  }),
  presentation: Object.freeze({
    yawDeg: 180,
    pivot: freezeVector([2.250915668599896, -3.375, 12.605660413500915]),
  }),
  camera: Object.freeze({
    preset: "reference",
    id: "pgsr_input_camera_000018",
    eye: freezeVector([-1.62950243820192, 1.818849050201853, -2.5396998253837135]),
    forward: freezeVector([0.31380919609254937, 0.005869813544282999, 0.9494679394693233]),
    up: freezeVector([-0.027003895871585824, 0.9996315591734393, 0.0027451343575030564]),
    targetDistance: 6,
    verticalFovDeg: 55.776917,
  }),
  robotSpawn: Object.freeze({
    id: "fixed-user-approved-interior-20260716",
    coordinateFrame: "visual_native",
    groundPoint: freezeVector([5.968559, 6.051313, 7.851893]),
    forward: freezeVector([0.31380919609254937, 0, 0.9494679394693233]),
  }),
});

export function baselinePresentation(manifest = {}) {
  const presentation = manifest.presentation;
  const pivot = Array.isArray(presentation?.pivot) && presentation.pivot.length === 3
    ? presentation.pivot.map(Number)
    : WEB_DEMO_BASELINE_252A85C.presentation.pivot;
  const yawDeg = Number.isFinite(Number(presentation?.yawDeg))
    ? Number(presentation.yawDeg)
    : WEB_DEMO_BASELINE_252A85C.presentation.yawDeg;
  return { pivot: [...pivot], yawDeg };
}
