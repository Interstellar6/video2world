import * as THREE from "three";
import { SplatMesh, SparkRenderer } from "@sparkjsdev/spark";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { acceleratedRaycast, MeshBVH, SAH } from "three-mesh-bvh";
import { baselinePresentation, WEB_DEMO_BASELINE_252A85C } from "./baseline-fixture.js";
import {
  answerSceneQuestion,
  buildSceneKnowledgeIndex,
  resolveSceneEntity,
} from "./scene-query.js";
import {
  buildSceneCommandEnvelope,
  parseSceneCommand,
  planSceneCommandEnvelope,
  submitSceneCommandEnvelope,
} from "./scene-command.js";
import {
  MAX_UNIFIED_GLTF_COLLISION_FACES,
  placementMatrixElements,
  validateWebManifest,
} from "./web-manifest.js";

const ASSET_VERSION = "video2world-scene-qa-object-colliders-v1";
const URL_PARAMS = new URLSearchParams(window.location.search);
const EXPLICIT_MANIFEST_URL = URL_PARAMS.get("manifest");
const MANIFEST_URL = EXPLICIT_MANIFEST_URL || "./test-fixtures/manifest.json";
const START_WITH_VISUAL = URL_PARAMS.get("visual") !== "off";
const START_WITH_STATIC_VISUAL = START_WITH_VISUAL && EXPLICIT_MANIFEST_URL != null;
const ALLOW_CANDIDATE_COLLIDERS = URL_PARAMS.get("allowCandidateColliders") === "1";
const START_WITH_COLLISION_DEBUG = URL_PARAMS.get("collisionDebug") === "on";
// Immutable mirrors of the exact chunk hashes referenced by this deployment.
let assetFetchSources = [
  { key: "origin", label: "origin", baseUrl: null, timeoutMs: 30_000 },
];
const ASSET_FETCH_CONCURRENCY = 3;
const ASSET_FETCH_RETRY_BASE_MS = 650;
let preferredAssetSourceKey = "origin";
const ALIGNMENT_STORAGE_KEY = "video2mesh-web-demo-alignment-pgsr-tsdf-v1";
const ALIGNMENT_NUDGE_STEP = 0.35;
const ALIGNMENT_ROTATION_STEP_DEG = 1.5;
const ALIGNMENT_ROTATION_LIMIT_DEG = { x: 45, y: 90, z: 45 };
const DEFAULT_SCENE_PRESENTATION_YAW_DEG = WEB_DEMO_BASELINE_252A85C.presentation.yawDeg;
const DEFAULT_SCENE_PRESENTATION_LABEL = "back-facing default";
const CAMERA_CONTROL_VERSION = "supersplat-free-orbit-20260716";
const DEFAULT_CAMERA_FOV_DEG = 58;
const CAMERA_ORBIT_SENSITIVITY = 0.0085;
const CAMERA_TRACKPAD_ORBIT_SENSITIVITY = 0.006;
const CAMERA_PAN_SENSITIVITY = 0.00145;
const CAMERA_WHEEL_ZOOM_SENSITIVITY = 0.00165;
const CAMERA_DRAG_ZOOM_SENSITIVITY = 0.012;
const CAMERA_MIN_POLAR = 0.035;
const CAMERA_MAX_POLAR = Math.PI - 0.035;
const CAMERA_PRESETS = ["reference", "doorway", "front", "back", "left", "right", "top", "robot"];
const CAMERA_PRESET_LABELS = {
  reference: "Reference interior",
  doorway: "Doorway entry",
  front: "Interior front",
  back: "Interior back",
  left: "Interior left",
  right: "Interior right",
  top: "Top down",
  robot: "Robot close",
  robotFollow: "Robot follow",
  overview: "Overview",
  custom: "Custom",
  fly: "Fly",
};
const COLLIDER_MODES = ["wire", "solid", "hidden"];
const COLLIDER_LABELS = {
  wire: "wire + xray",
  solid: "solid",
  hidden: "hidden but active",
};
const QUALITY_MODES = ["balanced", "performance"];
const QUALITY_LABELS = {
  balanced: "balanced",
  performance: "performance",
};
const ROBOT_GRAVITY = 12.5;
const ROBOT_JUMP_SPEED = 11.1;
const ROBOT_FALL_RESPAWN_DISTANCE = 8;
const PHYSICS_FIXED_STEP = 1 / 60;
const PHYSICS_MAX_FRAME_DELTA = 1;
const OBJECT_DRAG_YAW_PER_PIXEL = 0.01;
const ROBOT_UP = new THREE.Vector3(0, 1, 0);

const canvas = document.querySelector("#sceneCanvas");
const modeChip = document.querySelector("#modeChip");
const visualMetric = document.querySelector("#visualMetric");
const meshMetric = document.querySelector("#meshMetric");
const hitMetric = document.querySelector("#hitMetric");
const semanticMetric = document.querySelector("#semanticMetric");
const fpsMetric = document.querySelector("#fpsMetric");
const robotMetric = document.querySelector("#robotMetric");
const objectMetric = document.querySelector("#objectMetric");
const objectColliderMetric = document.querySelector("#objectColliderMetric");
const toast = document.querySelector("#toast");
const sceneQaForm = document.querySelector("#sceneQaForm");
const sceneQaInput = document.querySelector("#sceneQaInput");
const sceneQaAnswer = document.querySelector("#sceneQaAnswer");
const sceneQaCandidates = document.querySelector("#sceneQaCandidates");
const sceneCommandPreviewElement = document.querySelector("#sceneCommandPreview");
const sceneCommandTitle = document.querySelector("#sceneCommandTitle");
const sceneCommandRisk = document.querySelector("#sceneCommandRisk");
const sceneCommandSummary = document.querySelector("#sceneCommandSummary");
const sceneCommandStages = document.querySelector("#sceneCommandStages");
const sceneCommandStatus = document.querySelector("#sceneCommandStatus");
const sceneCommandCancel = document.querySelector("#sceneCommandCancel");
const sceneCommandConfirm = document.querySelector("#sceneCommandConfirm");

const nativePixelRatio = Math.max(0.75, window.devicePixelRatio || 1);
const initialPixelRatio = Math.min(nativePixelRatio, 1.35);
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: false,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(initialPixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x070a0d, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x070a0d, 18, 62);

const camera = new THREE.PerspectiveCamera(DEFAULT_CAMERA_FOV_DEG, window.innerWidth / window.innerHeight, 0.03, 180);
camera.position.set(24, 16, 30);

const controls = new OrbitControls(camera, canvas);
controls.enabled = false;
controls.enableDamping = false;
controls.dampingFactor = 0;
controls.enablePan = false;
controls.screenSpacePanning = true;
controls.enableZoom = false;
controls.enableRotate = false;
controls.rotateSpeed = 0;
controls.zoomSpeed = 0;
controls.panSpeed = 0;
controls.minDistance = 0.08;
controls.maxDistance = 220;
controls.maxPolarAngle = Math.PI;
controls.target.set(0, 0, 0);

const sparkRenderer = new SparkRenderer({ renderer, enableLod: true });
sparkRenderer.name = "spark-anysplat-visual-renderer";
scene.add(sparkRenderer);

scene.add(new THREE.HemisphereLight(0x9ff3ed, 0x18231f, 1.08));
scene.add(new THREE.AmbientLight(0xffffff, 1.15));
const sun = new THREE.DirectionalLight(0xffefd0, 1.8);
sun.position.set(8, 12, 7);
scene.add(sun);
const fill = new THREE.PointLight(0x58d7c9, 1.2, 80);
fill.position.set(-18, 8, -16);
scene.add(fill);

const visualLayer = new THREE.Group();
visualLayer.name = "visual-spark-3dgs-layer";
scene.add(visualLayer);

const colliderLayer = new THREE.Group();
colliderLayer.name = "mesh-collider-layer";
scene.add(colliderLayer);

const markerLayer = new THREE.Group();
markerLayer.name = "raycast-hit-marker-layer";
scene.add(markerLayer);

const robotLayer = new THREE.Group();
robotLayer.name = "mesh-ground-probe-robot-layer";
scene.add(robotLayer);

const interactiveObjectLayer = new THREE.Group();
interactiveObjectLayer.name = "interactive-gaussian-and-mesh-object-layer";
scene.add(interactiveObjectLayer);

const sceneEntityLayer = new THREE.Group();
sceneEntityLayer.name = "scene-knowledge-selection-layer";
interactiveObjectLayer.add(sceneEntityLayer);

const raycaster = new THREE.Raycaster();
const groundRaycaster = new THREE.Raycaster();
const obstacleRaycaster = new THREE.Raycaster();
const verticalRaycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const clock = new THREE.Clock();
const keyState = new Set();

const state = {
  manifestVersion: "",
  showVisual: START_WITH_VISUAL,
  showStaticVisual: true,
  colliderRenderMode: "hidden",
  semanticColor: true,
  cameraMode: "orbit",
  cameraPreset: "reference",
  cameraPresetSource: "pending",
  worldUp: { x: 0, y: 1, z: 0 },
  cameraControlVersion: CAMERA_CONTROL_VERSION,
  qualityMode: "balanced",
  targetPixelRatio: Number(initialPixelRatio.toFixed(2)),
  actualPixelRatio: Number(initialPixelRatio.toFixed(2)),
  adaptivePixelRatio: true,
  fps: 0,
  sampledWireFaces: 0,
  wireSampleStride: 1,
  visualReady: false,
  visualLoadSkipped: !START_WITH_STATIC_VISUAL,
  colliderReady: false,
  visualUsesSpark: true,
  visualAssetId: "",
  visualUrl: "",
  visualCount: 0,
  staticVisualCount: 0,
  visualSha256: "",
  visualSourcePath: "",
  colliderAssetId: "",
  colliderUrl: "",
  colliderFormat: "ply",
  colliderVertices: 0,
  colliderFaces: 0,
  colliderSha256: "",
  colliderSourcePath: "",
  colliderHasSemantics: false,
  colliderAcceleration: "pending",
  transformSource: "pending",
  transformScale: 1,
  transformTranslation: { x: 0, y: 0, z: 0 },
  transformOffset: { x: 0, y: 0, z: 0 },
  defaultViewerOffset: { x: 0, y: 0, z: 0 },
  scenePresentation: DEFAULT_SCENE_PRESENTATION_LABEL,
  scenePresentationYawDeg: DEFAULT_SCENE_PRESENTATION_YAW_DEG,
  scenePresentationPivot: { x: 0, y: 0, z: 0 },
  manualAlignmentOffset: { x: 0, y: 0, z: 0 },
  manualAlignmentRotationDeg: { x: 0, y: 0, z: 0 },
  transformRotationMatrix: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
  transformRmseSceneUnits: null,
  lastHit: "none",
  lastHitInfo: null,
  robotReady: false,
  robotEnabled: true,
  robotFollowCamera: false,
  robotGrounded: false,
  robotBlocked: false,
  robotObstacleCollision: true,
  robotVerticalSpeed: 0,
  robotJumpPhase: "grounded",
  robotAirTime: 0,
  robotJumpApex: 0,
  robotLastLanding: "spawn",
  robotSupportFace: null,
  physicsFrameDelta: 0,
  physicsSubsteps: 0,
  interactiveObjectCount: 0,
  interactiveObjectReadyCount: 0,
  interactiveObjectGaussianCount: 0,
  interactiveObjectRgbPointCount: 0,
  interactiveObjectMeshVertexCount: 0,
  interactiveObjectVisualPrimitiveCount: 0,
  interactiveObjectProxiesVisible: false,
  interactiveObjectSolo: false,
  selectedInteractiveObject: null,
  interactiveObjectRotationActive: false,
  interactiveObjectTurns: 0,
  activeObjectDrag: null,
  objectDragYawDeg: 0,
  objectColliderReadyCount: 0,
  objectColliderCount: 0,
  objectColliderMeshCount: 0,
  objectColliderFaces: 0,
  objectColliderLoadErrors: [],
  objectColliderDegradedCount: 0,
  objectColliderDebugVisible: START_WITH_COLLISION_DEBUG,
  objectColliderDebugObjectId: null,
  sceneQaReady: false,
  sceneQaQuery: "",
  sceneQaIntent: null,
  sceneQaStatus: "idle",
  sceneQaAnswer: "",
  sceneQaResolvedObjectId: null,
  sceneQaCandidates: [],
  sceneQaError: "",
  sceneCommandReady: false,
  sceneCommandKind: null,
  sceneCommandStatus: "idle",
  sceneCommandPreviewOnly: true,
  sceneCommandServiceAvailable: false,
  sceneCommandEndpoint: null,
  sceneCommandAffectedStages: [],
  sceneCommandTargetIds: [],
  sceneCommandRequestId: null,
  sceneCommandPlanStatus: "idle",
  sceneCommandConfirmationRequired: false,
  sceneCommandResponse: null,
  sceneCommandError: "",
  sceneEntityCount: 0,
  selectedSceneEntity: null,
  robotSupportColliderId: null,
  lastCollisionKind: null,
  lastCollisionObjectId: null,
  robotSpawnSource: "pending",
  robotFloorYEstimate: null,
  robotPosition: { x: 0, y: 0, z: 0 },
  robotYaw: 0,
  robotSpeed: 0,
  viewportGizmo: "idle",
  cameraPosition: { x: 0, y: 0, z: 0 },
  cameraTarget: { x: 0, y: 0, z: 0 },
  cameraDistance: 0,
  cameraOverviewFocusObjectId: null,
  cameraOverviewCoverage: null,
  cameraInsideInteractiveObjectIds: [],
  error: "",
};

let visualSplat = null;
let visualLoadPromise = null;
let interactiveObjectLoadPromise = null;
let interactiveObjectColliderLoadPromise = null;
let colliderMesh = null;
let colliderWire = null;
let robotGroup = null;
let robotGroundY = null;
let robotBobPhase = 0;
let robotVerticalSpeed = 0;
let robotAirTime = 0;
let robotJumpStartPosition = null;
let pointerDown = null;
let manifest = null;
let colliderFaceSemantics = [];
let colliderBounds = null;
let visualMetaBox = null;
let colliderMetaBox = null;
let transformedVisualBounds = null;
let baseVisualTransform = null;
let scenePresentationPivot = null;
let scenePresentationYawDeg = DEFAULT_SCENE_PRESENTATION_YAW_DEG;
let robotFloorYEstimate = null;
let lastFpsUpdate = 0;
let frameCount = 0;
let gizmoDrag = null;
let canvasDrag = null;
let objectDrag = null;
let sceneKnowledgeIndex = null;
let sceneEntitySelectionOutline = null;
let currentSceneCommandPreview = null;
let currentSceneCommandSubmission = null;
let sceneCommandEndpoint = null;
const interactiveObjects = new Map();
const interactiveObjectColliders = [];
const objectCollisionMeshes = [];

const smoothCamera = {
  initialized: false,
  target: new THREE.Vector3(),
  desiredTarget: new THREE.Vector3(),
  spherical: new THREE.Spherical(1, Math.PI / 2, 0),
  desiredSpherical: new THREE.Spherical(1, Math.PI / 2, 0),
  lastWheelTime: -Infinity,
  burstIsWheel: true,
};

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 1800);
}

function formatCount(value) {
  if (!Number.isFinite(value)) return "0";
  return Intl.NumberFormat("en-US", { notation: value >= 1000000 ? "compact" : "standard" }).format(value);
}

function nextValue(values, current) {
  const index = Math.max(0, values.indexOf(current));
  return values[(index + 1) % values.length];
}

function boxFromMeta(meta) {
  return new THREE.Box3(
    new THREE.Vector3(...meta.bbox.min),
    new THREE.Vector3(...meta.bbox.max)
  );
}

function boxSize(box) {
  return box.getSize(new THREE.Vector3());
}

function boxCenter(box) {
  return box.getCenter(new THREE.Vector3());
}

function semanticColor(objectId) {
  const palette = [
    0x5ad8c8, 0xefb35f, 0xff806d, 0x9fa8ff, 0x82d173,
    0xe681c9, 0x67b7ff, 0xd9cf73, 0xb48cff, 0xf28f5b,
  ];
  const id = Math.abs(Number(objectId) || 0);
  return new THREE.Color(palette[id % palette.length]);
}

function clampFinite(value, min, max, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return THREE.MathUtils.clamp(parsed, min, max);
}

function applyRendererPixelRatio(value, { forceResize = false } = {}) {
  const target = clampFinite(value, 0.75, Math.min(nativePixelRatio, 1.5), initialPixelRatio);
  const current = renderer.getPixelRatio();
  if (!forceResize && Math.abs(current - target) < 0.04) return;
  renderer.setPixelRatio(target);
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  state.actualPixelRatio = Number(renderer.getPixelRatio().toFixed(2));
  state.targetPixelRatio = Number(target.toFixed(2));
}

function qualityPixelRatioLimit() {
  return state.qualityMode === "performance"
    ? Math.min(nativePixelRatio, 1)
    : Math.min(nativePixelRatio, 1.35);
}

function applyQualityMode({ announce = false } = {}) {
  const limit = qualityPixelRatioLimit();
  state.adaptivePixelRatio = state.qualityMode !== "performance";
  applyRendererPixelRatio(limit, { forceResize: true });
  if (colliderMesh && state.qualityMode === "performance" && state.colliderRenderMode === "wire") {
    state.colliderRenderMode = "hidden";
    applyColliderMode();
  }
  if (announce) showToast(`Quality: ${QUALITY_LABELS[state.qualityMode]}, DPR ${state.actualPixelRatio}.`);
  updateHud();
}

function createSampledWireGeometry(geometry, maxFaces = 12000) {
  const index = geometry.index;
  const position = geometry.getAttribute("position");
  const faceCount = index ? Math.floor(index.count / 3) : Math.floor(position.count / 3);
  const stride = Math.max(1, Math.ceil(faceCount / maxFaces));
  const sampledFaces = Math.ceil(faceCount / stride);
  const linePositions = new Float32Array(sampledFaces * 18);
  let offset = 0;

  const writeVertex = (vertexIndex) => {
    linePositions[offset] = position.getX(vertexIndex);
    linePositions[offset + 1] = position.getY(vertexIndex);
    linePositions[offset + 2] = position.getZ(vertexIndex);
    offset += 3;
  };
  const writeEdge = (a, b) => {
    writeVertex(a);
    writeVertex(b);
  };

  for (let face = 0; face < faceCount; face += stride) {
    const base = face * 3;
    const a = index ? index.getX(base) : base;
    const b = index ? index.getX(base + 1) : base + 1;
    const c = index ? index.getX(base + 2) : base + 2;
    writeEdge(a, b);
    writeEdge(b, c);
    writeEdge(c, a);
  }

  const wireGeometry = new THREE.BufferGeometry();
  wireGeometry.setAttribute("position", new THREE.BufferAttribute(linePositions.subarray(0, offset), 3));
  state.sampledWireFaces = sampledFaces;
  state.wireSampleStride = stride;
  return wireGeometry;
}

function identityRotationRows() {
  return [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
}

function rotationRowsToMatrix4(rows) {
  return new THREE.Matrix4().set(
    rows[0][0], rows[0][1], rows[0][2], 0,
    rows[1][0], rows[1][1], rows[1][2], 0,
    rows[2][0], rows[2][1], rows[2][2], 0,
    0, 0, 0, 1
  );
}

function normalizeRotationRows(rows) {
  if (!Array.isArray(rows) || rows.length !== 3) return identityRotationRows();
  const parsed = rows.map((row) => Array.isArray(row) ? row.map(Number) : []);
  if (parsed.some((row) => row.length !== 3 || row.some((value) => !Number.isFinite(value)))) {
    return identityRotationRows();
  }
  return parsed;
}

function vectorFromArray(values, fallback = [0, 0, 0]) {
  const source = Array.isArray(values) && values.length >= 3 ? values : fallback;
  return new THREE.Vector3(Number(source[0]) || 0, Number(source[1]) || 0, Number(source[2]) || 0);
}

function roundedVector(vector) {
  return {
    x: Number(vector.x.toFixed(6)),
    y: Number(vector.y.toFixed(6)),
    z: Number(vector.z.toFixed(6)),
  };
}

function roundedEuler(euler) {
  return {
    x: Number(euler.x.toFixed(4)),
    y: Number(euler.y.toFixed(4)),
    z: Number(euler.z.toFixed(4)),
  };
}

function roundedRotationRows(rows) {
  return rows.map((row) => row.map((value) => Number(Number(value).toFixed(8))));
}

function serializeBox(box) {
  if (!box || box.isEmpty()) return null;
  return {
    min: roundedVector(box.min),
    max: roundedVector(box.max),
    center: roundedVector(boxCenter(box)),
    size: roundedVector(boxSize(box)),
  };
}

function currentVisualTransform() {
  return {
    source: state.transformSource,
    scale: state.transformScale,
    rotationMatrix: state.transformRotationMatrix,
    translation: [
      state.transformTranslation.x,
      state.transformTranslation.y,
      state.transformTranslation.z,
    ],
    manualAlignmentOffset: [
      state.manualAlignmentOffset.x,
      state.manualAlignmentOffset.y,
      state.manualAlignmentOffset.z,
    ],
    defaultViewerOffset: [
      state.defaultViewerOffset.x,
      state.defaultViewerOffset.y,
      state.defaultViewerOffset.z,
    ],
    manualAlignmentRotationDeg: [
      state.manualAlignmentRotationDeg.x,
      state.manualAlignmentRotationDeg.y,
      state.manualAlignmentRotationDeg.z,
    ],
    scenePresentation: state.scenePresentation,
    scenePresentationYawDeg: state.scenePresentationYawDeg,
    scenePresentationPivot: [
      state.scenePresentationPivot.x,
      state.scenePresentationPivot.y,
      state.scenePresentationPivot.z,
    ],
    transformedVisualBounds: serializeBox(transformedVisualBounds),
    rmseSceneUnits: state.transformRmseSceneUnits,
  };
}

function rotationRowsFromEulerDeg(degrees) {
  const euler = new THREE.Euler(
    THREE.MathUtils.degToRad(Number(degrees?.[0]) || 0),
    THREE.MathUtils.degToRad(Number(degrees?.[1]) || 0),
    THREE.MathUtils.degToRad(Number(degrees?.[2]) || 0),
    "XYZ"
  );
  const elements = new THREE.Matrix4().makeRotationFromEuler(euler).elements;
  return [
    [elements[0], elements[4], elements[8]],
    [elements[1], elements[5], elements[9]],
    [elements[2], elements[6], elements[10]],
  ];
}

function matrix4ToRotationRows(matrix) {
  const elements = matrix.elements;
  return [
    [elements[0], elements[4], elements[8]],
    [elements[1], elements[5], elements[9]],
    [elements[2], elements[6], elements[10]],
  ];
}

function cloneTransform(transform) {
  return {
    source: transform.source,
    scale: Number(transform.scale),
    rotationMatrix: normalizeRotationRows(transform.rotationMatrix),
    translation: transform.translation.clone(),
    rmseSceneUnits: transform.rmseSceneUnits,
    maxErrorSceneUnits: transform.maxErrorSceneUnits,
  };
}

function calculateTransformedBox(box, transform) {
  if (!box || !transform) return null;
  const rotation4 = rotationRowsToMatrix4(normalizeRotationRows(transform.rotationMatrix));
  const quaternion = new THREE.Quaternion().setFromRotationMatrix(rotation4);
  const matrix = new THREE.Matrix4().compose(
    transform.translation,
    quaternion,
    new THREE.Vector3(transform.scale, transform.scale, transform.scale)
  );
  return box.clone().applyMatrix4(matrix);
}

function scenePresentationRotationMatrix4() {
  return rotationRowsToMatrix4(rotationRowsFromEulerDeg([0, scenePresentationYawDeg, 0]));
}

function scenePresentationMatrix4(pivot = scenePresentationPivot) {
  if (!pivot) return new THREE.Matrix4().identity();
  const rotation = scenePresentationRotationMatrix4();
  return new THREE.Matrix4()
    .makeTranslation(pivot.x, pivot.y, pivot.z)
    .multiply(rotation)
    .multiply(new THREE.Matrix4().makeTranslation(-pivot.x, -pivot.y, -pivot.z));
}

function applyScenePresentationToGeometry(geometry, pivot) {
  if (!geometry || !pivot) return;
  geometry.applyMatrix4(scenePresentationMatrix4(pivot));
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
}

function applyScenePresentationToVisualTransform(transform) {
  if (!transform || !scenePresentationPivot) return transform;
  const presented = cloneTransform(transform);
  const rotation = scenePresentationRotationMatrix4();
  const baseRotation = rotationRowsToMatrix4(presented.rotationMatrix);
  const combinedRotation = rotation.clone().multiply(baseRotation);
  const pivotShift = scenePresentationPivot.clone()
    .sub(scenePresentationPivot.clone().applyMatrix4(rotation));

  presented.rotationMatrix = matrix4ToRotationRows(combinedRotation);
  presented.translation = presented.translation.clone().applyMatrix4(rotation).add(pivotShift);
  presented.source = `${presented.source}+${DEFAULT_SCENE_PRESENTATION_LABEL.replace(/\s+/g, "_")}`;
  return presented;
}

function calculateVisualToColliderTransform(visualBox, colliderBox, alignmentMeta = null) {
  if (alignmentMeta?.rotationMatrix && Number.isFinite(Number(alignmentMeta.scale))) {
    const rotationMatrix = normalizeRotationRows(alignmentMeta.rotationMatrix);
    const translation = vectorFromArray(alignmentMeta.translation);
    const defaultViewerOffset = vectorFromArray(alignmentMeta.viewerDefaultOffset);
    translation.add(defaultViewerOffset);
    state.defaultViewerOffset = roundedVector(defaultViewerOffset);
    const source = alignmentMeta.viewerDefaultOffset
      ? `${alignmentMeta.id || alignmentMeta.method || "manifest_alignment"}+viewer_default`
      : (alignmentMeta.id || alignmentMeta.method || "manifest_alignment");
    return {
      source,
      scale: Number(alignmentMeta.scale),
      rotationMatrix,
      translation,
      rmseSceneUnits: Number.isFinite(Number(alignmentMeta.rmseSceneUnits))
        ? Number(alignmentMeta.rmseSceneUnits)
        : null,
      maxErrorSceneUnits: Number.isFinite(Number(alignmentMeta.maxErrorSceneUnits))
        ? Number(alignmentMeta.maxErrorSceneUnits)
        : null,
    };
  }
  const visualSize = boxSize(visualBox);
  const colliderSize = boxSize(colliderBox);
  const visualMax = Math.max(visualSize.x, visualSize.y, visualSize.z, 1e-6);
  const colliderMax = Math.max(colliderSize.x, colliderSize.y, colliderSize.z, 1e-6);
  const scale = colliderMax / visualMax;
  const visualCenter = boxCenter(visualBox);
  const colliderCenter = boxCenter(colliderBox);
  const translation = colliderCenter.clone().sub(visualCenter.clone().multiplyScalar(scale));
  state.defaultViewerOffset = { x: 0, y: 0, z: 0 };
  return {
    source: "bbox_center_max_extent_fallback",
    scale,
    rotationMatrix: identityRotationRows(),
    translation,
    rmseSceneUnits: null,
    maxErrorSceneUnits: null,
  };
}

function sanitizeAlignmentVector(value) {
  const source = value && typeof value === "object" ? value : {};
  return {
    x: clampFinite(source.x, -12, 12, 0),
    y: clampFinite(source.y, -12, 12, 0),
    z: clampFinite(source.z, -12, 12, 0),
  };
}

function sanitizeAlignmentEuler(value) {
  const source = value && typeof value === "object" ? value : {};
  return {
    x: clampFinite(source.x, -ALIGNMENT_ROTATION_LIMIT_DEG.x, ALIGNMENT_ROTATION_LIMIT_DEG.x, 0),
    y: clampFinite(source.y, -ALIGNMENT_ROTATION_LIMIT_DEG.y, ALIGNMENT_ROTATION_LIMIT_DEG.y, 0),
    z: clampFinite(source.z, -ALIGNMENT_ROTATION_LIMIT_DEG.z, ALIGNMENT_ROTATION_LIMIT_DEG.z, 0),
  };
}

function loadManualAlignment() {
  try {
    const raw = localStorage.getItem(ALIGNMENT_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    state.manualAlignmentOffset = sanitizeAlignmentVector(parsed.offset);
    state.manualAlignmentRotationDeg = sanitizeAlignmentEuler(parsed.rotationDeg);
  } catch (error) {
    console.warn("Failed to load viewer alignment offset", error);
    state.manualAlignmentOffset = { x: 0, y: 0, z: 0 };
    state.manualAlignmentRotationDeg = { x: 0, y: 0, z: 0 };
  }
}

function saveManualAlignment() {
  const payload = {
    offset: state.manualAlignmentOffset,
    rotationDeg: state.manualAlignmentRotationDeg,
    version: ASSET_VERSION,
  };
  localStorage.setItem(ALIGNMENT_STORAGE_KEY, JSON.stringify(payload));
}

function manualAlignmentIsActive() {
  const offset = state.manualAlignmentOffset;
  const rotation = state.manualAlignmentRotationDeg;
  return (
    Math.abs(offset.x) > 1e-4 ||
    Math.abs(offset.y) > 1e-4 ||
    Math.abs(offset.z) > 1e-4 ||
    Math.abs(rotation.x) > 1e-4 ||
    Math.abs(rotation.y) > 1e-4 ||
    Math.abs(rotation.z) > 1e-4
  );
}

function buildEffectiveVisualTransform() {
  if (!baseVisualTransform) return null;
  const base = applyScenePresentationToVisualTransform(baseVisualTransform);
  const manualOffset = new THREE.Vector3(
    state.manualAlignmentOffset.x,
    state.manualAlignmentOffset.y,
    state.manualAlignmentOffset.z
  );
  const manualRows = rotationRowsFromEulerDeg([
    state.manualAlignmentRotationDeg.x,
    state.manualAlignmentRotationDeg.y,
    state.manualAlignmentRotationDeg.z,
  ]);
  const manualMatrix = rotationRowsToMatrix4(manualRows);
  const baseMatrix = rotationRowsToMatrix4(base.rotationMatrix);
  const combinedMatrix = manualMatrix.clone().multiply(baseMatrix);

  const baseBox = calculateTransformedBox(visualMetaBox, base);
  const pivot = baseBox ? boxCenter(baseBox) : base.translation.clone();
  const adjustedTranslation = base.translation.clone().sub(pivot).applyMatrix4(manualMatrix).add(pivot).add(manualOffset);
  const active = manualAlignmentIsActive();
  return {
    source: active ? `${base.source}+viewer_adjust` : base.source,
    scale: base.scale,
    rotationMatrix: matrix4ToRotationRows(combinedMatrix),
    translation: adjustedTranslation,
    rmseSceneUnits: base.rmseSceneUnits,
    maxErrorSceneUnits: base.maxErrorSceneUnits,
  };
}

function applyVisualTransform(transform, { announce = false } = {}) {
  const rotationMatrix = normalizeRotationRows(transform.rotationMatrix);
  const rotation4 = rotationRowsToMatrix4(rotationMatrix);
  const applyToLayer = (layer) => {
    layer.scale.setScalar(transform.scale);
    layer.quaternion.setFromRotationMatrix(rotation4);
    layer.position.copy(transform.translation);
    layer.updateMatrixWorld(true);
  };
  if (visualSplat) applyToLayer(visualSplat);
  applyToLayer(interactiveObjectLayer);
  transformedVisualBounds = calculateTransformedBox(visualMetaBox, transform);

  state.transformSource = transform.source || "manual";
  state.transformScale = Number(transform.scale.toFixed(8));
  state.transformTranslation = roundedVector(transform.translation);
  state.transformOffset = state.transformTranslation;
  state.transformRotationMatrix = roundedRotationRows(rotationMatrix);
  state.transformRmseSceneUnits = Number.isFinite(transform.rmseSceneUnits)
    ? Number(transform.rmseSceneUnits.toFixed(6))
    : null;
  updateHud();
  if (announce) showToast(`Visual alignment: ${state.transformSource}`);
}

function applyEffectiveVisualTransform({ announce = false } = {}) {
  const transform = buildEffectiveVisualTransform();
  if (!transform) return null;
  applyVisualTransform(transform, { announce });
  return transform;
}

function nudgeVisualAlignment(delta) {
  const current = new THREE.Vector3(
    state.manualAlignmentOffset.x,
    state.manualAlignmentOffset.y,
    state.manualAlignmentOffset.z
  );
  current.add(delta);
  state.manualAlignmentOffset = sanitizeAlignmentVector(current);
  saveManualAlignment();
  applyEffectiveVisualTransform();
  updateAlignmentButtons();
  showToast(`Visual offset ${state.manualAlignmentOffset.x.toFixed(2)}, ${state.manualAlignmentOffset.y.toFixed(2)}, ${state.manualAlignmentOffset.z.toFixed(2)}.`);
}

function nudgeVisualRotation(axis, deltaDeg) {
  if (!["x", "y", "z"].includes(axis)) return;
  const next = {
    ...state.manualAlignmentRotationDeg,
    [axis]: state.manualAlignmentRotationDeg[axis] + deltaDeg,
  };
  state.manualAlignmentRotationDeg = sanitizeAlignmentEuler(next);
  saveManualAlignment();
  applyEffectiveVisualTransform();
  updateAlignmentButtons();
  showToast(`Visual rotation ${axis.toUpperCase()} ${state.manualAlignmentRotationDeg[axis].toFixed(1)} deg.`);
}

function resetManualAlignment() {
  state.manualAlignmentOffset = { x: 0, y: 0, z: 0 };
  state.manualAlignmentRotationDeg = { x: 0, y: 0, z: 0 };
  saveManualAlignment();
  applyEffectiveVisualTransform({ announce: true });
  updateAlignmentButtons();
  showToast("Viewer alignment reset to default manifest correction.");
}

function buildManualVisualTransform(patch = {}) {
  const rotationMatrix = patch.rotationMatrix
    ? normalizeRotationRows(patch.rotationMatrix)
    : (patch.rotationEulerDeg ? rotationRowsFromEulerDeg(patch.rotationEulerDeg) : normalizeRotationRows(state.transformRotationMatrix));

  return {
    source: patch.source || "manual_viewer_alignment",
    scale: Number.isFinite(Number(patch.scale)) ? Number(patch.scale) : state.transformScale,
    rotationMatrix,
    translation: patch.translation
      ? vectorFromArray(patch.translation)
      : new THREE.Vector3(
        state.transformTranslation.x,
        state.transformTranslation.y,
        state.transformTranslation.z
      ),
    rmseSceneUnits: null,
    maxErrorSceneUnits: null,
  };
}

function exposeDebugApi() {
  window.__setCameraPreset = (preset = "reference") => {
    setCameraOrbitView(preset, { announce: false, immediate: true });
    return syncDebugState();
  };
  window.__setVisualAlignment = (patch = {}) => {
    const transform = buildManualVisualTransform(patch);
    baseVisualTransform = transform;
    applyVisualTransform(transform, { announce: true });
    return syncDebugState();
  };
  window.__getVisualAlignment = () => currentVisualTransform();
  window.__nudgeVisualAlignment = (delta = {}) => {
    nudgeVisualAlignment(new THREE.Vector3(
      Number(delta.x) || 0,
      Number(delta.y) || 0,
      Number(delta.z) || 0
    ));
    return syncDebugState();
  };
  window.__resetVisualAlignmentOffset = () => {
    resetManualAlignment();
    return syncDebugState();
  };
  window.__jumpRobot = () => {
    startRobotJump();
    return syncDebugState();
  };
  window.__probeWalkableSurface = (x, z) => {
    const probeY = robotGroup?.position.y ?? robotFloorYEstimate ?? 0;
    const hit = groundProbe(new THREE.Vector3(Number(x), probeY, Number(z)), { preferFloor: false });
    if (!hit) return null;
    const normal = worldHitNormal(hit);
    return {
      point: roundedVector(hit.point),
      normal: roundedVector(normal),
      faceIndex: Number.isInteger(hit.faceIndex) ? hit.faceIndex : null,
      colliderKind: hit.object.userData.colliderKind || "scene",
      colliderId: hit.object.userData.colliderId || "scene",
      objectId: hit.object.userData.interactiveObjectId || null,
    };
  };
  window.__spinInteractiveObject = (objectId) => {
    const component = interactiveObjects.get(String(objectId));
    return { started: startInteractiveObjectSpin(component), state: syncDebugState() };
  };
  window.__loadInteractiveObjectVisuals = async () => {
    await ensureInteractiveObjectsLoaded();
    return syncDebugState();
  };
  window.__focusInteractiveObject = (objectId) => {
    const component = interactiveObjects.get(String(objectId));
    if (component) selectInteractiveObject(component, { focus: true });
    return syncDebugState();
  };
  window.__askScene = (question) => askScene(question);
  window.__previewSceneCommand = (question) => previewSceneInput(question);
  window.__confirmSceneCommand = () => submitCurrentSceneCommand();
  window.__cancelSceneCommand = () => {
    clearSceneCommandPreview();
    updateHud();
    return syncDebugState();
  };
  window.__resolveSceneEntity = (question) => {
    if (!sceneKnowledgeIndex) return null;
    const result = resolveSceneEntity(question, sceneKnowledgeIndex, {
      selectedEntityId: state.selectedSceneEntity,
    });
    return {
      status: result.status,
      entityId: result.entity?.id || null,
      candidates: (result.candidates || []).map((entity) => entity.id),
    };
  };
  window.__focusSceneEntity = (objectId) => ({
    focused: focusSceneEntity(String(objectId), { focus: true }),
    state: syncDebugState(),
  });
  window.__setInteractiveObjectYaw = (objectId, degrees) => ({
    updated: setInteractiveObjectYaw(interactiveObjects.get(String(objectId)), degrees),
    state: syncDebugState(),
  });
  window.__getInteractiveObjectScreenPoint = (objectId) => {
    const component = interactiveObjects.get(String(objectId));
    if (!component) return null;
    component.group.updateMatrixWorld(true);
    const collisionBounds = interactiveObjectCollisionBounds(component);
    const worldPoint = collisionBounds && !collisionBounds.isEmpty()
      ? collisionBounds.getCenter(new THREE.Vector3())
      : component.group.getWorldPosition(new THREE.Vector3());
    const projected = worldPoint.project(camera);
    const rect = canvas.getBoundingClientRect();
    return {
      x: rect.left + (projected.x * 0.5 + 0.5) * rect.width,
      y: rect.top + (-projected.y * 0.5 + 0.5) * rect.height,
      ndcZ: projected.z,
    };
  };
  window.__inspectInteractiveObjectHitAt = (clientX, clientY) => {
    const hit = interactiveObjectHitAt(clientX, clientY);
    if (!hit) return null;
    return {
      objectId: hit.object.userData.interactiveObjectId || null,
      colliderMode: hit.object.userData.colliderMode || null,
      collisionTopology: hit.object.userData.collisionTopology || null,
    };
  };
  window.__inspectCollisionWorld = () => ({
    sceneReady: Boolean(colliderMesh),
    sceneFaces: state.colliderFaces,
    targetCount: collisionTargets().length,
    objectColliderReadyCount: state.objectColliderReadyCount,
    objectColliderMeshCount: state.objectColliderMeshCount,
    objectColliderFaces: state.objectColliderFaces,
    degradedCount: state.objectColliderDegradedCount,
    errors: [...state.objectColliderLoadErrors],
    objects: Array.from(interactiveObjects.values()).map((component) => ({
      id: component.definition.id,
      ready: component.colliderReady,
      mode: component.collisionMode,
      gateStatus: component.definition.collision?.gate?.status || "missing",
      topology: component.definition.collision?.topology || null,
      volumePhysics: component.definition.collision?.topology === "closed_volume",
      walkable: component.collisionMeshes.some((mesh) => mesh.userData.walkable === true),
      characterCollision: component.collisionMeshes.some(
        (mesh) => mesh.userData.characterCollision === true,
      ),
      faces: component.collisionFaces,
      bounds: serializeBox(interactiveObjectCollisionBounds(component)),
    })),
  });
  window.__inspectSceneInterpenetrations = () => inspectSceneInterpenetrations();
  window.__setObjectColliderDebugVisible = (visible, objectId = null) => ({
    visible: setObjectColliderDebugVisible(visible, objectId),
    objectId: state.objectColliderDebugObjectId,
    state: syncDebugState(),
  });
  window.__probeCollisionRay = (origin, direction, far = 100) => {
    const rayOrigin = vectorFromArray(origin);
    const rayDirection = vectorFromArray(direction);
    if (rayDirection.lengthSq() < 1e-8) return null;
    const probe = new THREE.Raycaster(rayOrigin, rayDirection.normalize(), 0, Number(far) || 100);
    const hit = intersectCollisionWorld(probe)[0];
    if (!hit) return null;
    return describeHit(hit, worldHitNormal(hit));
  };
  window.__probeRobotCollision = (from, to) => {
    const start = vectorFromArray(from);
    const end = vectorFromArray(to);
    const direction = end.clone().sub(start);
    const distance = direction.length();
    if (distance < 1e-8) return null;
    const probe = new THREE.Raycaster(start, direction.normalize(), 0, distance);
    const hit = intersectCollisionWorld(probe, { characterCollision: true })[0];
    return hit ? describeHit(hit, worldHitNormal(hit)) : null;
  };
  window.__prepareRobotCollisionTest = (objectId) => {
    const component = interactiveObjects.get(String(objectId));
    const bounds = component ? interactiveObjectCollisionBounds(component) : null;
    const characterCollision = component?.collisionMeshes.some(
      (mesh) => mesh.userData.characterCollision === true,
    );
    if (!component || !bounds || !component.colliderReady || !colliderMesh) {
      return { prepared: false, reason: "object_or_collider_not_ready", state: syncDebugState() };
    }
    if (!characterCollision) {
      return { prepared: false, reason: "character_collision_not_approved", state: syncDebugState() };
    }

    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const roomCenter = colliderBounds
      ? colliderBounds.getCenter(new THREE.Vector3())
      : new THREE.Vector3();
    const outward = center.clone().sub(roomCenter);
    outward.addScaledVector(ROBOT_UP, -outward.dot(ROBOT_UP));
    if (outward.lengthSq() < 1e-6) {
      camera.getWorldDirection(outward).multiplyScalar(-1);
      outward.addScaledVector(ROBOT_UP, -outward.dot(ROBOT_UP));
    }
    if (outward.lengthSq() < 1e-6) outward.set(1, 0, 0);
    outward.normalize();

    const candidateDirections = [
      outward,
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(-1, 0, 0),
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(0, 0, -1),
    ];
    let surfaceHit = null;
    let surfaceOutward = null;
    const verticalExtent = Math.abs(size.dot(ROBOT_UP));
    for (const heightFraction of [0, -0.38, 0.38, -0.22, 0.22]) {
      if (surfaceHit) break;
      const sampleCenter = center.clone().addScaledVector(ROBOT_UP, verticalExtent * heightFraction);
      for (const candidate of candidateDirections) {
        const candidateOutward = candidate.clone();
        candidateOutward.addScaledVector(ROBOT_UP, -candidateOutward.dot(ROBOT_UP));
        if (candidateOutward.lengthSq() < 1e-8) continue;
        candidateOutward.normalize();
        const rayOrigin = sampleCenter.clone().addScaledVector(candidateOutward, Math.max(2, size.length() + 1));
        const surfaceRay = new THREE.Raycaster(
          rayOrigin,
          candidateOutward.clone().multiplyScalar(-1),
          0,
          size.length() + 4
        );
        surfaceHit = intersectCollisionWorld(surfaceRay, { characterCollision: true })
          .find((hit) => (
            hit.object.userData.interactiveObjectId === component.definition.id
            && Math.abs(worldHitNormal(hit).dot(ROBOT_UP)) < 0.52
          ));
        if (surfaceHit) {
          surfaceOutward = candidateOutward;
          break;
        }
      }
    }
    if (!surfaceHit) {
      return { prepared: false, reason: "no_outward_surface_hit", state: syncDebugState() };
    }

    const outsidePoint = surfaceHit.point.clone().addScaledVector(surfaceOutward, 0.22);
    let support = groundProbe(outsidePoint, { preferFloor: true });
    let supportSource = "walkable_scene_surface";
    if (support) respawnRobotAt(support.point, `collision-test:${component.definition.id}`);

    const pointsAtObject = () => {
      if (!robotGroup) return false;
      const direction = center.clone().sub(robotGroup.position);
      direction.addScaledVector(ROBOT_UP, -direction.dot(ROBOT_UP));
      if (direction.lengthSq() < 1e-8) return false;
      const origin = robotGroup.position.clone().addScaledVector(ROBOT_UP, 0.58);
      const probe = new THREE.Raycaster(origin, direction.normalize(), 0.04, 0.72);
      const hit = intersectCollisionWorld(probe, { characterCollision: true })
        .find((candidate) => Math.abs(worldHitNormal(candidate).dot(ROBOT_UP)) < 0.52);
      return hit?.object.userData.interactiveObjectId === component.definition.id;
    };

    if (!support || !pointsAtObject()) {
      const harnessSupport = outsidePoint
        .clone()
        .addScaledVector(ROBOT_UP, -(0.58 + (robotGroup?.userData.groundOffset || 0.02)));
      if (!respawnRobotAt(harnessSupport, `collision-harness:${component.definition.id}`)) {
        return { prepared: false, reason: "unable_to_place_collision_harness", state: syncDebugState() };
      }
      support = { point: harnessSupport };
      supportSource = "object_midplane_collision_harness";
    }
    if (!pointsAtObject()) {
      return { prepared: false, reason: "robot_obstacle_ray_does_not_target_object", state: syncDebugState() };
    }
    state.lastCollisionKind = null;
    state.lastCollisionObjectId = null;
    state.robotBlocked = false;
    const stateSnapshot = syncDebugState();
    return {
      prepared: true,
      objectId: component.definition.id,
      outward: roundedVector(surfaceOutward),
      surfacePoint: roundedVector(surfaceHit.point),
      supportPoint: roundedVector(support.point),
      supportSource,
      robotPosition: stateSnapshot.robotPosition,
      state: stateSnapshot,
    };
  };
  window.__stepRobotTowardObject = (objectId, distance = 0.65) => {
    const component = interactiveObjects.get(String(objectId));
    const bounds = component ? interactiveObjectCollisionBounds(component) : null;
    if (!component || !bounds || !robotGroup) {
      return { attempted: false, reason: "object_or_robot_not_ready", state: syncDebugState() };
    }
    const before = robotGroup.position.clone();
    const direction = bounds.getCenter(new THREE.Vector3()).sub(before);
    direction.addScaledVector(ROBOT_UP, -direction.dot(ROBOT_UP));
    if (direction.lengthSq() < 1e-8) {
      return { attempted: false, reason: "robot_at_object_center", state: syncDebugState() };
    }
    const requestedDistance = THREE.MathUtils.clamp(Number(distance) || 0.65, 0.05, 2);
    const moved = moveRobotOnMesh(direction.normalize(), requestedDistance);
    const stateSnapshot = syncDebugState();
    return {
      attempted: true,
      moved,
      blocked: stateSnapshot.robotBlocked,
      lastCollisionKind: stateSnapshot.lastCollisionKind,
      lastCollisionObjectId: stateSnapshot.lastCollisionObjectId,
      before: roundedVector(before),
      after: stateSnapshot.robotPosition,
      requestedDistance,
      state: stateSnapshot,
    };
  };
  window.__inspectInteractiveObject = (objectId) => {
    const component = interactiveObjects.get(String(objectId));
    if (!component) return null;
    component.group.updateMatrixWorld(true);
    const splatLocalBounds = component.splat?.getBoundingBox?.(true) || component.splatLocalBounds;
    const splatWorldBounds = splatLocalBounds && component.splat
      ? splatLocalBounds.clone().applyMatrix4(component.splat.matrixWorld)
      : null;
    const proxyLocalBounds = component.proxy
      ? (component.proxy.geometry.boundingBox
        || (component.proxy.geometry.computeBoundingBox(), component.proxy.geometry.boundingBox))
      : null;
    const proxyWorldBounds = proxyLocalBounds
      ? proxyLocalBounds.clone().applyMatrix4(component.proxy.matrixWorld)
      : null;
    const describeBox = (box) => ({
      min: roundedVector(box.min),
      max: roundedVector(box.max),
      center: roundedVector(box.getCenter(new THREE.Vector3())),
      size: roundedVector(box.getSize(new THREE.Vector3())),
    });
    const collisionMaterials = component.collisionMeshes.flatMap((mesh) => (
      Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    )).filter(Boolean);
    return {
      id: component.definition.id,
      parentObjectId: component.definition.parentObjectId,
      movesWithParent: component.definition.movesWithParent,
      independentlyMovable: component.definition.independentlyMovable,
      childObjectIds: Array.from(component.childComponents, (child) => child.definition.id),
      hierarchyAttachmentMatrixDelta: component.hierarchyAttachmentMatrixDelta,
      visualReady: component.visualReady,
      colliderReady: component.colliderReady,
      visualKind: component.visualKind,
      collisionMode: component.collisionMode,
      interactionKind: component.definition.interaction?.kind || null,
      interactionDrag: component.definition.interaction?.drag || null,
      collisionGateStatus: component.definition.collision?.gate?.status || "missing",
      collisionTopology: component.definition.collision?.topology || null,
      unifiedVisualCollision: component.collisionMode === "unified-glb"
        && component.splat === component.collisionRoot,
      walkable: component.collisionMeshes.some((mesh) => mesh.userData.walkable === true),
      characterCollision: component.collisionMeshes.some(
        (mesh) => mesh.userData.characterCollision === true,
      ),
      bvhMeshCount: component.collisionMeshes.filter((mesh) => mesh.geometry?.boundsTree).length,
      materialTypes: [...new Set(collisionMaterials.map((material) => material.type))],
      pbrMaterialCount: collisionMaterials.filter(
        (material) => material.isMeshStandardMaterial || material.isMeshPhysicalMaterial,
      ).length,
      collisionFaces: component.collisionFaces,
      collisionBounds: interactiveObjectCollisionBounds(component)
        ? describeBox(interactiveObjectCollisionBounds(component))
        : null,
      selectionOutlineVisible: component.outline.visible === true,
      turns: component.turns,
      meshVertexCount: component.meshVisualVertexCount,
      splatLocalBounds: splatLocalBounds ? describeBox(splatLocalBounds) : null,
      splatWorldBounds: splatWorldBounds ? describeBox(splatWorldBounds) : null,
      proxyWorldBounds: proxyWorldBounds ? describeBox(proxyWorldBounds) : null,
      groupMatrix: component.group.matrix.elements.map((value) => Number(value.toFixed(6))),
      groupMatrixWorld: component.group.matrixWorld.elements.map((value) => Number(value.toFixed(6))),
      splatMatrixWorld: component.splat
        ? component.splat.matrixWorld.elements.map((value) => Number(value.toFixed(6)))
        : null,
      collisionMatrixWorld: component.collisionMeshes[0]
        ? component.collisionMeshes[0].matrixWorld.elements.map((value) => Number(value.toFixed(6)))
        : null,
      proxyMatrixWorld: component.proxy
        ? component.proxy.matrixWorld.elements.map((value) => Number(value.toFixed(6)))
        : null,
    };
  };
}

function clampCameraSpherical(spherical) {
  spherical.radius = THREE.MathUtils.clamp(
    spherical.radius,
    Math.max(0.02, controls.minDistance || 0.02),
    Math.max(0.08, controls.maxDistance || 220)
  );
  spherical.phi = THREE.MathUtils.clamp(spherical.phi, CAMERA_MIN_POLAR, CAMERA_MAX_POLAR);
  return spherical;
}

function sphericalFromCamera(eye = camera.position, target = controls.target) {
  return clampCameraSpherical(new THREE.Spherical().setFromVector3(eye.clone().sub(target)));
}

function syncSmoothCameraFromCurrent({ immediate = true } = {}) {
  const spherical = sphericalFromCamera(camera.position, controls.target);
  smoothCamera.desiredTarget.copy(controls.target);
  smoothCamera.desiredSpherical.copy(spherical);
  if (immediate || !smoothCamera.initialized) {
    smoothCamera.target.copy(controls.target);
    smoothCamera.spherical.copy(spherical);
  }
  smoothCamera.initialized = true;
}

function applySmoothCameraPose() {
  clampCameraSpherical(smoothCamera.spherical);
  const offset = new THREE.Vector3().setFromSpherical(smoothCamera.spherical);
  camera.position.copy(smoothCamera.target).add(offset);
  controls.target.copy(smoothCamera.target);
  camera.lookAt(smoothCamera.target);
}

function setSmoothCameraPose(eye, target, { immediate = false } = {}) {
  const shouldSnap = immediate || !smoothCamera.initialized;
  smoothCamera.desiredTarget.copy(target);
  smoothCamera.desiredSpherical.copy(sphericalFromCamera(eye, target));
  smoothCamera.initialized = true;
  if (shouldSnap) {
    smoothCamera.target.copy(smoothCamera.desiredTarget);
    smoothCamera.spherical.copy(smoothCamera.desiredSpherical);
    applySmoothCameraPose();
  }
}

function setSmoothCameraTarget(target, { preserveOffset = true, immediate = false } = {}) {
  if (!smoothCamera.initialized) syncSmoothCameraFromCurrent();
  smoothCamera.desiredTarget.copy(target);
  if (!preserveOffset) smoothCamera.desiredSpherical.copy(sphericalFromCamera(camera.position, target));
  if (immediate) {
    smoothCamera.target.copy(smoothCamera.desiredTarget);
    applySmoothCameraPose();
  }
}

function orbitCameraAroundTarget({ theta = 0, phi = 0, scale = 1, announce = false } = {}) {
  if (state.cameraMode !== "orbit") state.cameraMode = "orbit";
  if (!smoothCamera.initialized) syncSmoothCameraFromCurrent();
  smoothCamera.desiredSpherical.theta += theta;
  smoothCamera.desiredSpherical.phi += phi;
  smoothCamera.desiredSpherical.radius *= scale;
  clampCameraSpherical(smoothCamera.desiredSpherical);
  state.robotFollowCamera = false;
  state.cameraPreset = "custom";
  state.cameraPresetSource = "free-orbit";
  if (announce) setLayerVisibility();
  if (announce) showToast("Camera adjusted around the current indoor target.");
}

function panCameraTarget(deltaX, deltaY, { announce = false } = {}) {
  if (!smoothCamera.initialized) syncSmoothCameraFromCurrent();
  const distance = Math.max(smoothCamera.desiredSpherical.radius, 0.1);
  const panScale = distance * CAMERA_PAN_SENSITIVITY;
  const eye = smoothCamera.desiredTarget.clone().add(new THREE.Vector3().setFromSpherical(smoothCamera.desiredSpherical));
  const forward = smoothCamera.desiredTarget.clone().sub(eye).normalize();
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  const up = new THREE.Vector3().crossVectors(right, forward).normalize();
  const move = new THREE.Vector3()
    .addScaledVector(right, -deltaX * panScale)
    .addScaledVector(up, deltaY * panScale);
  smoothCamera.desiredTarget.add(move);
  smoothCamera.target.add(move.multiplyScalar(0.22));
  state.robotFollowCamera = false;
  state.cameraMode = "orbit";
  state.cameraPreset = "custom";
  state.cameraPresetSource = "free-orbit-pan";
  if (announce) setLayerVisibility();
  if (announce) showToast("Camera target panned.");
}

function zoomCameraByWheel(deltaY, sensitivity = CAMERA_WHEEL_ZOOM_SENSITIVITY) {
  if (!smoothCamera.initialized) syncSmoothCameraFromCurrent();
  const scale = Math.exp(THREE.MathUtils.clamp(deltaY * sensitivity, -0.7, 0.7));
  orbitCameraAroundTarget({ scale });
}

function moveCameraRigAlongView(distance, { announce = false } = {}) {
  if (!smoothCamera.initialized) syncSmoothCameraFromCurrent();
  const eye = smoothCamera.desiredTarget.clone()
    .add(new THREE.Vector3().setFromSpherical(smoothCamera.desiredSpherical));
  const forward = smoothCamera.desiredTarget.clone().sub(eye).normalize();
  const move = forward.multiplyScalar(distance);
  smoothCamera.desiredTarget.add(move);
  smoothCamera.target.add(move.clone().multiplyScalar(0.22));
  state.robotFollowCamera = false;
  state.cameraMode = "orbit";
  state.cameraPreset = "custom";
  state.cameraPresetSource = "free-orbit-dolly";
  if (announce) setLayerVisibility();
  if (announce) showToast(distance >= 0 ? "Camera moved into the room." : "Camera moved toward the doorway.");
}

function classifyWheelInput(event) {
  if (event.deltaMode !== WheelEvent.DOM_DELTA_PIXEL) return true;
  const rawEvent = event;
  if (typeof rawEvent.wheelDeltaY === "number" && rawEvent.wheelDeltaY !== 0) {
    return rawEvent.wheelDeltaY % 120 === 0;
  }
  if (typeof rawEvent.wheelDeltaX === "number" && rawEvent.wheelDeltaX !== 0) {
    return rawEvent.wheelDeltaX % 120 === 0;
  }
  if (event.deltaX !== 0 && event.deltaY !== 0) return false;
  return Number.isInteger(event.deltaX) && Number.isInteger(event.deltaY);
}

function updateSmoothCamera(dt) {
  if (state.cameraMode !== "orbit") return;
  if (!smoothCamera.initialized) syncSmoothCameraFromCurrent();
  const targetAlpha = 1 - Math.exp(-Math.max(dt, 0.001) * 16);
  const angleAlpha = 1 - Math.exp(-Math.max(dt, 0.001) * 18);
  const radiusAlpha = 1 - Math.exp(-Math.max(dt, 0.001) * 20);

  smoothCamera.target.lerp(smoothCamera.desiredTarget, targetAlpha);
  const thetaDelta = Math.atan2(
    Math.sin(smoothCamera.desiredSpherical.theta - smoothCamera.spherical.theta),
    Math.cos(smoothCamera.desiredSpherical.theta - smoothCamera.spherical.theta)
  );
  smoothCamera.spherical.theta += thetaDelta * angleAlpha;
  smoothCamera.spherical.phi += (smoothCamera.desiredSpherical.phi - smoothCamera.spherical.phi) * angleAlpha;
  smoothCamera.spherical.radius += (smoothCamera.desiredSpherical.radius - smoothCamera.spherical.radius) * radiusAlpha;
  applySmoothCameraPose();
}

function syncCameraState() {
  interactiveObjectLayer.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  state.cameraPosition = roundedVector(camera.position);
  state.cameraTarget = roundedVector(controls.target);
  state.cameraDistance = Number(camera.position.distanceTo(controls.target).toFixed(4));
  state.cameraInsideInteractiveObjectIds = Array.from(interactiveObjects.values())
    .filter((component) => interactiveObjectCollisionBounds(component)?.containsPoint(camera.position))
    .map((component) => component.definition.id);
  const overview = interactiveObjects.get(state.cameraOverviewFocusObjectId);
  const overviewBounds = overview ? interactiveObjectCollisionBounds(overview) : null;
  if (overviewBounds && !overviewBounds.isEmpty()) {
    const points = [];
    for (const x of [overviewBounds.min.x, overviewBounds.max.x]) {
      for (const y of [overviewBounds.min.y, overviewBounds.max.y]) {
        for (const z of [overviewBounds.min.z, overviewBounds.max.z]) {
          points.push(new THREE.Vector3(x, y, z).project(camera));
        }
      }
    }
    const minX = Math.min(...points.map((point) => point.x));
    const maxX = Math.max(...points.map((point) => point.x));
    const minY = Math.min(...points.map((point) => point.y));
    const maxY = Math.max(...points.map((point) => point.y));
    state.cameraOverviewCoverage = {
      widthFraction: Number(((maxX - minX) * 0.5).toFixed(4)),
      heightFraction: Number(((maxY - minY) * 0.5).toFixed(4)),
      minimumNdcZ: Number(Math.min(...points.map((point) => point.z)).toFixed(4)),
      maximumNdcZ: Number(Math.max(...points.map((point) => point.z)).toFixed(4)),
    };
  } else {
    state.cameraOverviewCoverage = null;
  }
}

function syncDebugState() {
  syncCameraState();
  const interactiveObjectState = Array.from(interactiveObjects.values()).map((component) => {
    const splatWorldBounds = component.splatLocalBounds && component.splat
      ? component.splatLocalBounds.clone().applyMatrix4(component.splat.matrixWorld)
      : null;
    const proxyLocalBounds = component.proxy?.geometry.boundingBox || null;
    const proxyWorldBounds = proxyLocalBounds
      ? proxyLocalBounds.clone().applyMatrix4(component.proxy.matrixWorld)
      : null;
    return {
      id: component.definition.id,
      label: component.definition.label,
      semanticGranularity: component.definition.semanticGranularity,
      parentObjectId: component.definition.parentObjectId,
      movesWithParent: component.definition.movesWithParent,
      independentlyMovable: component.definition.independentlyMovable,
      interactionKind: component.definition.interaction?.kind || null,
      interactionDrag: component.definition.interaction?.drag || null,
      childObjectIds: Array.from(component.childComponents, (child) => child.definition.id),
      ready: component.ready,
      visualReady: component.visualReady,
      colliderReady: component.colliderReady,
      visualKind: component.visualKind,
      collisionMode: component.collisionMode,
      collisionGateStatus: component.definition.collision?.gate?.status || "missing",
      collisionTopology: component.definition.collision?.topology || null,
      unifiedVisualCollision: component.collisionMode === "unified-glb"
        && component.splat === component.collisionRoot,
      walkable: component.collisionMeshes.some((mesh) => mesh.userData.walkable === true),
      characterCollision: component.collisionMeshes.some(
        (mesh) => mesh.userData.characterCollision === true,
      ),
      collisionMeshCount: component.collisionMeshes.length,
      collisionFaces: component.collisionFaces,
      collisionBounds: serializeBox(interactiveObjectCollisionBounds(component)),
      selectionOutlineVisible: component.outline.visible === true,
      spinning: Boolean(component.spin),
      dragging: objectDrag?.component === component,
      turns: component.turns,
      gaussianCount: component.definition.visual?.vertexCount || 0,
      meshVertexCount: component.meshVisualVertexCount,
      position: roundedVector(component.group.getWorldPosition(new THREE.Vector3())),
      proxyDimensions: component.definition.colliderProxy?.dimensions || null,
      splatWorldBounds: serializeBox(splatWorldBounds),
      proxyWorldBounds: serializeBox(proxyWorldBounds),
      groupMatrixWorld: component.group.matrixWorld.elements.map((value) => Number(value.toFixed(6))),
      splatMatrixWorld: component.splat
        ? component.splat.matrixWorld.elements.map((value) => Number(value.toFixed(6)))
        : null,
      proxyMatrixWorld: component.proxy
        ? component.proxy.matrixWorld.elements.map((value) => Number(value.toFixed(6)))
        : null,
    };
  });
  const snapshot = {
    ...state,
    sparkRendererVisible: sparkRenderer.visible,
    visualLayerVisible: visualLayer.visible,
    colliderLayerVisible: colliderLayer.visible,
    colliderVisible: state.colliderRenderMode !== "hidden",
    rendererPixelRatio: Number(renderer.getPixelRatio().toFixed(3)),
    cameraFovDeg: Number(camera.fov.toFixed(6)),
    robotVisible: Boolean(robotGroup?.visible),
    colliderBounds: serializeBox(colliderBounds),
    visualMetaBounds: serializeBox(visualMetaBox),
    transformedVisualBounds: serializeBox(transformedVisualBounds),
    collisionTargetCount: collisionTargets().length,
    interactiveObjects: interactiveObjectState,
  };
  window.__visualPhysicsDemoState = snapshot;
  document.documentElement.dataset.visualPhysicsState = JSON.stringify(snapshot);
  return snapshot;
}

function updateHud() {
  visualMetric.textContent = state.visualReady
    ? formatCount(state.visualCount)
    : state.visualLoadSkipped ? "off" : "loading";
  meshMetric.textContent = state.colliderReady ? formatCount(state.colliderFaces) : "loading";
  hitMetric.textContent = state.lastHitInfo ? `face ${state.lastHitInfo.faceIndex}` : "none";
  semanticMetric.textContent = state.lastHitInfo
    ? (state.lastHitInfo.objectId == null ? "n/a" : `id ${state.lastHitInfo.objectId}`)
    : "none";
  robotMetric.textContent = state.robotReady
    ? (state.robotBlocked
      ? "blocked"
      : state.robotGrounded
        ? `${state.robotSpeed.toFixed(1)} u/s`
        : `${state.robotJumpPhase} ${Math.abs(state.robotVerticalSpeed).toFixed(1)}`)
    : "loading";
  objectMetric.textContent = state.interactiveObjectCount
    ? `${state.interactiveObjectReadyCount}/${state.interactiveObjectCount}`
    : "0";
  objectColliderMetric.textContent = state.objectColliderCount
    ? `${state.objectColliderReadyCount}/${state.objectColliderCount}`
    : "0";
  modeChip.textContent = [
    state.visualReady
      ? "PGSR Gaussian PLY visual ready"
      : state.visualLoadSkipped ? "visual PLY deferred" : "loading visual PLY",
    state.interactiveObjectCount
      ? `visual objects ${state.interactiveObjectReadyCount}/${state.interactiveObjectCount}`
      : "no visual objects",
    state.objectColliderCount
      ? `object colliders ${state.objectColliderReadyCount}/${state.objectColliderCount}`
      : "no object colliders",
    state.colliderReady ? "TSDF PLY collider ready" : "loading collider PLY",
    state.transformSource === "pending"
      ? "alignment pending"
      : `${state.transformSource}${state.transformRmseSceneUnits ? ` rmse ${state.transformRmseSceneUnits}` : ""}`,
    `${state.scenePresentation} yaw ${state.scenePresentationYawDeg}°`,
    manualAlignmentIsActive()
      ? `manual offset ${state.manualAlignmentOffset.x.toFixed(2)}/${state.manualAlignmentOffset.y.toFixed(2)}/${state.manualAlignmentOffset.z.toFixed(2)} yaw ${state.manualAlignmentRotationDeg.y.toFixed(1)}`
      : (state.transformSource.includes("viewer_default") ? "bbox-center viewer correction" : "manifest alignment"),
    `${COLLIDER_LABELS[state.colliderRenderMode]} collider`,
    state.colliderAcceleration,
    `${CAMERA_PRESET_LABELS[state.cameraPreset] || state.cameraPreset} ${state.cameraMode} camera`,
    `${QUALITY_LABELS[state.qualityMode]} dpr ${state.actualPixelRatio}`,
    state.robotEnabled
      ? `robot ${state.robotObstacleCollision ? "collides with mesh" : "ground-only"} ${state.robotJumpPhase} ${state.robotSpawnSource}`
      : "robot hidden",
    "raycast ignores visual splats",
  ].join(" · ");
  syncDebugState();
}

async function loadManifest() {
  const response = await fetch(MANIFEST_URL);
  if (!response.ok) throw new Error(`Manifest failed: ${response.status} ${response.statusText}`);
  manifest = validateWebManifest(await response.json());
  state.manifestVersion = manifest.version || "";
  if (Array.isArray(manifest.assetFetchSources)) {
    const configuredSources = manifest.assetFetchSources
      .filter((source) => source && typeof source === "object" && source.baseUrl)
      .map((source, index) => ({
        key: String(source.key || `mirror-${index + 1}`),
        label: String(source.label || source.key || `mirror ${index + 1}`),
        baseUrl: String(source.baseUrl),
        timeoutMs: clampFinite(source.timeoutMs, 5_000, 120_000, 75_000),
      }));
    assetFetchSources = [assetFetchSources[0], ...configuredSources];
  }
  const presentation = baselinePresentation(manifest);
  scenePresentationPivot = vectorFromArray(presentation.pivot);
  scenePresentationYawDeg = presentation.yawDeg;
  state.scenePresentationPivot = roundedVector(scenePresentationPivot);
  state.scenePresentationYawDeg = scenePresentationYawDeg;
  const configuredUp = vectorFromArray(manifest.coordinateSystem?.worldUp);
  ROBOT_UP.copy(configuredUp);
  state.worldUp = roundedVector(ROBOT_UP);
  camera.up.copy(ROBOT_UP);
  const initialState = manifest.initialState || {};
  if (manifest.cameraPresets?.[initialState.cameraPreset]) {
    state.cameraPreset = initialState.cameraPreset;
  }
  if (typeof initialState.robotObstacleCollision === "boolean") {
    state.robotObstacleCollision = initialState.robotObstacleCollision;
  }
  state.interactiveObjectCount = Array.isArray(manifest.interactiveObjects)
    ? manifest.interactiveObjects.length
    : 0;
  state.objectColliderCount = Array.isArray(manifest.interactiveObjects)
    ? manifest.interactiveObjects.filter((definition) => definition.collision?.mode !== "none").length
    : 0;
  sceneKnowledgeIndex = buildSceneKnowledgeIndex(manifest);
  sceneCommandEndpoint = manifest.sceneCommandService?.endpoint
    ? new URL(manifest.sceneCommandService.endpoint, new URL(MANIFEST_URL, window.location.href)).href
    : null;
  state.sceneCommandServiceAvailable = Boolean(sceneCommandEndpoint);
  state.sceneCommandEndpoint = sceneCommandEndpoint;
  state.sceneEntityCount = sceneKnowledgeIndex.entities.size;
  state.sceneQaReady = true;
  return manifest;
}

function getAssetPartSources(part) {
  const path = part.url.replace(/^\.\//, "");
  const sources = assetFetchSources.map((source) => ({
    ...source,
    url: source.baseUrl ? `${source.baseUrl}${path}` : `${part.url}?v=${ASSET_VERSION}`,
  }));
  const preferred = sources.find((source) => source.key === preferredAssetSourceKey) || sources[0];
  if (preferred.key === "origin") return sources;
  return [
    preferred,
    ...sources.filter((source) => source.key !== preferred.key && source.key !== "origin"),
    sources[0],
  ];
}

async function sha256Hex(bytes) {
  if (!globalThis.crypto?.subtle) throw new Error("Web Crypto SHA-256 is unavailable");
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
}

async function fetchPart(part, label, onLoaded) {
  const sources = getAssetPartSources(part);
  let lastFailure = "unknown error";
  for (let attempt = 0; attempt < sources.length; attempt += 1) {
    const source = sources[attempt];
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), source.timeoutMs);
    try {
      const response = await fetch(source.url, {
        cache: "force-cache",
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (part.size && bytes.length !== part.size) {
        throw new Error(`size mismatch: expected ${part.size}, got ${bytes.length}`);
      }
      if (part.sha256) {
        const actualSha256 = await sha256Hex(bytes);
        if (actualSha256 !== part.sha256) {
          throw new Error(`SHA-256 mismatch: expected ${part.sha256}, got ${actualSha256}`);
        }
      }
      preferredAssetSourceKey = source.key;
      onLoaded?.(bytes.length);
      return bytes;
    } catch (error) {
      lastFailure = error?.name === "AbortError"
        ? `${source.label} timed out after ${source.timeoutMs / 1000}s`
        : (error?.message || String(error));
      const nextSource = sources[attempt + 1];
      if (!nextSource) break;
      preferredAssetSourceKey = nextSource.key;
      const fileName = part.url.split("/").pop() || part.url;
      modeChip.textContent = `${label} retry ${attempt + 1}/${sources.length - 1} via ${nextSource.label} · ${fileName}`;
      await new Promise((resolve) => window.setTimeout(resolve, ASSET_FETCH_RETRY_BASE_MS * (attempt + 1)));
    } finally {
      window.clearTimeout(timeoutId);
    }
  }
  throw new Error(`${label} chunk failed across ${sources.length} sources: ${part.url} (${lastFailure})`);
}

async function getChunkedAssetBytes(asset, assetKey = "asset") {
  if (!asset?.parts?.length) {
    const url = asset?.url || asset?.fileName;
    if (!url) throw new Error(`No chunk list or URL for ${assetKey}`);
    let bytes = await fetchPart({
      url: String(url).startsWith(".") ? url : `./assets/${url}`,
      size: asset.transportSize,
      sha256: asset.transportSha256,
    }, asset.label || assetKey);
    if (asset.encoding === "base64") {
      const encoded = new TextDecoder().decode(bytes).trim();
      const binary = window.atob(encoded);
      bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    }
    if (Number.isFinite(Number(asset.size)) && bytes.length !== Number(asset.size)) {
      throw new Error(`${assetKey} size mismatch: expected ${asset.size}, got ${bytes.length}`);
    }
    if (asset.sha256) {
      const actualSha256 = await sha256Hex(bytes);
      if (actualSha256 !== asset.sha256) {
        throw new Error(`${assetKey} SHA-256 mismatch: expected ${asset.sha256}, got ${actualSha256}`);
      }
    }
    return { asset, bytes };
  }
  const merged = new Uint8Array(asset.size);
  const offsets = [];
  let expectedTotal = 0;
  for (const part of asset.parts) {
    offsets.push(expectedTotal);
    expectedTotal += Number(part.size) || 0;
  }
  if (expectedTotal !== asset.size) {
    throw new Error(`${assetKey} manifest size mismatch: expected ${asset.size}, parts sum ${expectedTotal}`);
  }
  let loaded = 0;
  let nextIndex = 0;
  const label = asset.label || assetKey;
  const concurrency = Math.min(asset.parts.length, ASSET_FETCH_CONCURRENCY);
  const onLoaded = (length) => {
    loaded += length;
    const pct = asset.size ? Math.round((loaded / asset.size) * 100) : 0;
    modeChip.textContent = `${label} chunks ${pct}%`;
  };
  async function worker() {
    while (nextIndex < asset.parts.length) {
      const index = nextIndex;
      nextIndex += 1;
      const bytes = await fetchPart(asset.parts[index], label, onLoaded);
      merged.set(bytes, offsets[index]);
    }
  }
  await Promise.all(Array.from({ length: concurrency }, () => worker()));
  return { asset, bytes: merged };
}

async function getChunkedBytes(assetKey) {
  return getChunkedAssetBytes(manifest.assets[assetKey], assetKey);
}

async function loadVisualSplat(transform) {
  const { asset, bytes } = await getChunkedBytes("visual");
  state.visualAssetId = asset.id;
  state.visualUrl = asset.fileName;
  state.staticVisualCount = asset.vertexCount || 0;
  state.visualCount = state.staticVisualCount + state.interactiveObjectVisualPrimitiveCount;
  state.visualSha256 = asset.sha256 || "";
  state.visualSourcePath = asset.sourcePath || "";
  state.transformSource = transform.source;
  state.transformScale = transform.scale;
  state.transformTranslation = roundedVector(transform.translation);
  state.transformOffset = state.transformTranslation;
  state.transformRotationMatrix = roundedRotationRows(transform.rotationMatrix);
  state.transformRmseSceneUnits = Number.isFinite(transform.rmseSceneUnits)
    ? Number(transform.rmseSceneUnits.toFixed(6))
    : null;

  visualSplat = new SplatMesh({
    fileBytes: bytes,
    fileName: asset.fileName,
    fileType: asset.fileType,
    lod: true,
    raycastable: false,
    onProgress: (event) => {
      if (!event.total) return;
      const pct = Math.round((event.loaded / event.total) * 100);
      modeChip.textContent = `Spark decode ${pct}%`;
    },
  });
  visualSplat.name = asset.label || "PGSR bedroom_4 Gaussian visual proxy";
  visualSplat.raycast = () => {};
  visualLayer.add(visualSplat);
  applyVisualTransform(transform);

  await visualSplat.initialized;
  applyVisualTransform(transform);
  state.visualReady = true;
  showToast("PGSR Gaussian PLY visual layer loaded.");
  updateHud();
  setLayerVisibility();
}

function createInteractiveObjectComponent(definition) {
  const unified = definition.collision?.mode === "unified-glb";
  const group = new THREE.Group();
  group.name = `${definition.id} interactive visual-collider component`;
  const configuredPlacementMatrix = placementMatrixElements(definition.placement);
  const configuredPlacement = configuredPlacementMatrix
    ? new THREE.Matrix4().fromArray(configuredPlacementMatrix)
    : null;
  if (configuredPlacementMatrix) {
    group.position.setFromMatrixPosition(configuredPlacement);
  } else {
    group.position.copy(vectorFromArray(definition.placement?.pivot));
  }
  group.userData.interactiveObjectId = definition.id;

  const content = new THREE.Group();
  content.name = `${definition.id} canonical Gaussian placement`;
  if (configuredPlacement) {
    const ignoredTranslation = new THREE.Vector3();
    configuredPlacement.decompose(ignoredTranslation, content.quaternion, content.scale);
  } else {
    const rotation = definition.placement?.rotationEulerDeg || [0, 0, 0];
    content.rotation.set(
      THREE.MathUtils.degToRad(Number(rotation[0]) || 0),
      THREE.MathUtils.degToRad(Number(rotation[1]) || 0),
      THREE.MathUtils.degToRad(Number(rotation[2]) || 0),
      definition.placement?.eulerOrder || "XYZ"
    );
    content.scale.copy(vectorFromArray(definition.placement?.scale, [1, 1, 1]));
  }
  group.add(content);

  let proxy = null;
  let proxyGeometry = new THREE.BufferGeometry();
  if (!unified) {
    const dimensions = vectorFromArray(definition.colliderProxy?.dimensions, [0.5, 0.5, 0.5]);
    proxyGeometry = new THREE.BoxGeometry(dimensions.x, dimensions.y, dimensions.z);
    proxyGeometry.computeBoundingBox();
    const proxyMaterial = new THREE.MeshBasicMaterial({
      color: 0xefb35f,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    proxy = new THREE.Mesh(proxyGeometry, proxyMaterial);
    proxy.name = `${definition.id} robust-bounds mesh proxy`;
    proxy.position.copy(vectorFromArray(definition.colliderProxy?.center));
    proxy.userData.interactiveObjectId = definition.id;
    proxy.userData.colliderLabel = `${definition.label} interactive box proxy`;
    proxy.userData.surfaceType = "interactive-object-box-proxy";
    group.add(proxy);
  }
  const outline = new THREE.LineSegments(
    unified ? proxyGeometry : new THREE.EdgesGeometry(proxyGeometry),
    new THREE.LineBasicMaterial({ color: 0x58d7c9, transparent: true, opacity: 0.92 })
  );
  outline.name = `${definition.id} selection outline`;
  if (proxy) outline.position.copy(proxy.position);
  outline.visible = !unified && state.interactiveObjectProxiesVisible;
  outline.raycast = () => {};
  group.add(outline);

  const visualAsset = unified ? definition.collision.asset : definition.visual;
  const component = {
    definition,
    group,
    content,
    splat: null,
    visualKind: isObjectMeshVisual(visualAsset)
      ? "mesh"
      : visualAsset?.renderer === "three-points" ? "rgb-points" : "gaussian-splat",
    proxy,
    outline,
    ready: false,
    visualReady: false,
    colliderReady: false,
    collisionSettled: false,
    collisionMode: "pending",
    collisionRoot: null,
    collisionMeshes: [],
    collisionFaces: 0,
    collisionBounds: null,
    splatLocalBounds: definition.visual?.bbox ? boxFromMeta(definition.visual) : null,
    meshVisualVertexCount: 0,
    spin: null,
    turns: 0,
    dragYawRadians: 0,
    restQuaternion: group.quaternion.clone(),
    parentComponent: null,
    childComponents: new Set(),
    hierarchyAttachmentMatrixDelta: 0,
    hierarchyResolved: false,
    unifiedLoadPromise: null,
  };
  interactiveObjects.set(definition.id, component);
  if (proxy) interactiveObjectColliders.push(proxy);
  interactiveObjectLayer.add(group);
  return component;
}

function isObjectMeshVisual(asset) {
  const descriptor = [asset?.fileName, asset?.fileType, asset?.format]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return /(?:^|[.\s-])glb(?:$|[.\s-])|gltf-binary|(?:^|[.\s-])obj(?:$|[.\s-])|wavefront/u
    .test(descriptor);
}

function configureUnifiedSelectionOutline(component, localBounds) {
  const size = localBounds.getSize(new THREE.Vector3());
  const center = localBounds.getCenter(new THREE.Vector3());
  const box = new THREE.BoxGeometry(
    Math.max(size.x, 1e-5),
    Math.max(size.y, 1e-5),
    Math.max(size.z, 1e-5),
  );
  const edges = new THREE.EdgesGeometry(box);
  box.dispose();
  component.outline.geometry.dispose();
  component.outline.geometry = edges;
  component.outline.position.copy(center);
  component.content.add(component.outline);
}

function prepareUnifiedMeshGeometry(root, component) {
  const meshes = [];
  let faceCount = 0;
  let vertexCount = 0;
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  root.traverse((node) => {
    if (node.isSkinnedMesh) {
      throw new Error("unified-glb does not support skinned collision geometry");
    }
    if (!node.isMesh) return;
    const geometry = node.geometry;
    const positions = geometry?.getAttribute?.("position");
    if (!positions?.count) throw new Error("unified-glb mesh is missing vertex positions");
    const index = geometry.index;
    const elementCount = index ? index.count : positions.count;
    if (elementCount % 3 !== 0) throw new Error("unified-glb mesh is not triangulated");
    for (let vertex = 0; vertex < positions.count; vertex += 1) {
      if (
        !Number.isFinite(positions.getX(vertex))
        || !Number.isFinite(positions.getY(vertex))
        || !Number.isFinite(positions.getZ(vertex))
      ) throw new Error("unified-glb mesh contains non-finite positions");
    }
    for (let offset = 0; offset < elementCount; offset += 3) {
      const ia = index ? index.getX(offset) : offset;
      const ib = index ? index.getX(offset + 1) : offset + 1;
      const ic = index ? index.getX(offset + 2) : offset + 2;
      if (ia >= positions.count || ib >= positions.count || ic >= positions.count) {
        throw new Error("unified-glb mesh contains out-of-range triangle indices");
      }
      a.fromBufferAttribute(positions, ia);
      b.fromBufferAttribute(positions, ib);
      c.fromBufferAttribute(positions, ic);
      ab.subVectors(b, a);
      ac.subVectors(c, a);
      if (ab.cross(ac).lengthSq() <= 1e-20) {
        throw new Error("unified-glb mesh contains degenerate triangles");
      }
    }
    if (!geometry.getAttribute("normal")) geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    if (!geometry.boundsTree) {
      geometry.boundsTree = new MeshBVH(geometry, {
        strategy: SAH,
        indirect: true,
        maxLeafTris: 12,
      });
    }
    node.raycast = acceleratedRaycast;
    node.frustumCulled = false;
    node.userData.interactiveObjectId = component.definition.id;
    node.userData.colliderId = component.definition.id;
    node.userData.colliderKind = "object";
    node.userData.colliderMode = "unified-glb";
    node.userData.collisionTopology = component.definition.collision.topology;
    node.userData.volumePhysics = component.definition.collision.topology === "closed_volume";
    node.userData.collisionGateStatus = "passed";
    node.userData.colliderLabel = `${component.definition.label} unified PBR mesh`;
    node.userData.surfaceType = "interactive-object-unified-glb";
    node.userData.walkable = component.definition.collision.walkable === true;
    node.userData.characterCollision = true;
    meshes.push(node);
    vertexCount += positions.count;
    faceCount += elementCount / 3;
  });
  if (!meshes.length) throw new Error("unified-glb contains no triangle meshes");
  if (faceCount > MAX_UNIFIED_GLTF_COLLISION_FACES) {
    throw new Error(`unified-glb exceeds ${MAX_UNIFIED_GLTF_COLLISION_FACES} collision faces`);
  }
  if (faceCount !== component.definition.collision.asset.faces) {
    throw new Error(
      `unified-glb face count ${faceCount} does not match manifest ${component.definition.collision.asset.faces}`,
    );
  }
  return { meshes, faceCount, vertexCount };
}

async function loadUnifiedInteractiveObject(component) {
  if (component.ready && component.collisionSettled) return component;
  if (component.unifiedLoadPromise) return component.unifiedLoadPromise;
  component.unifiedLoadPromise = (async () => {
    const definition = component.definition;
    try {
      const { bytes } = await getChunkedAssetBytes(definition.collision.asset, definition.id);
      const loaded = await parseObjectCollision(bytes, definition.collision.asset);
      loaded.name = `${definition.label} unified PBR visual and collider`;
      const prepared = prepareUnifiedMeshGeometry(loaded, component);
      loaded.updateMatrixWorld(true);
      const contentBounds = new THREE.Box3().setFromObject(loaded);
      if (contentBounds.isEmpty()) throw new Error("unified-glb has empty bounds");
      const rootLocalBounds = contentBounds.clone().applyMatrix4(
        loaded.matrixWorld.clone().invert(),
      );
      configureUnifiedSelectionOutline(component, contentBounds);
      loaded.visible = state.showVisual;
      component.content.add(loaded);
      component.splat = loaded;
      component.splatLocalBounds = rootLocalBounds;
      component.meshVisualVertexCount = prepared.vertexCount;
      component.collisionRoot = loaded;
      component.collisionMeshes = prepared.meshes;
      component.collisionFaces = prepared.faceCount;
      component.collisionMode = "unified-glb";
      component.collisionSettled = true;
      component.colliderReady = true;
      component.visualKind = "mesh";
      component.visualReady = true;
      component.ready = true;
      for (const mesh of prepared.meshes) {
        objectCollisionMeshes.push(mesh);
        interactiveObjectColliders.push(mesh);
      }
      component.group.updateMatrixWorld(true);
      component.collisionBounds = interactiveObjectCollisionBounds(component);
      state.objectColliderReadyCount += 1;
      state.objectColliderMeshCount += prepared.meshes.length;
      state.objectColliderFaces += prepared.faceCount;
      state.interactiveObjectReadyCount += 1;
      state.interactiveObjectMeshVertexCount += prepared.vertexCount;
      state.interactiveObjectVisualPrimitiveCount = state.interactiveObjectGaussianCount
        + state.interactiveObjectRgbPointCount
        + state.interactiveObjectMeshVertexCount;
      state.visualCount = state.staticVisualCount + state.interactiveObjectVisualPrimitiveCount;
      updateHud();
      return component;
    } catch (error) {
      component.ready = false;
      component.visualReady = false;
      component.colliderReady = false;
      component.collisionSettled = true;
      component.collisionMode = "none";
      component.collisionError = error?.message || String(error);
      state.objectColliderLoadErrors.push({ id: definition.id, error: component.collisionError });
      throw new Error(`${definition.id}: unified-glb rejected: ${component.collisionError}`);
    }
  })();
  try {
    return await component.unifiedLoadPromise;
  } finally {
    component.unifiedLoadPromise = null;
  }
}

function applyInteractiveObjectHierarchy() {
  interactiveObjectLayer.updateMatrixWorld(true);
  for (const component of interactiveObjects.values()) {
    if (component.hierarchyResolved) continue;
    const parentObjectId = component.definition.parentObjectId;
    if (parentObjectId != null) {
      const parent = interactiveObjects.get(parentObjectId);
      if (!parent) throw new Error(`${component.definition.id}: missing parent component ${parentObjectId}`);
      component.group.updateMatrixWorld(true);
      const worldBeforeAttach = component.group.matrixWorld.clone();
      parent.group.attach(component.group);
      component.group.updateMatrixWorld(true);
      component.hierarchyAttachmentMatrixDelta = Math.max(
        ...worldBeforeAttach.elements.map((value, index) => (
          Math.abs(value - component.group.matrixWorld.elements[index])
        )),
      );
      component.parentComponent = parent;
      parent.childComponents.add(component);
      component.restQuaternion.copy(component.group.quaternion);
    }
    component.hierarchyResolved = true;
  }
  interactiveObjectLayer.updateMatrixWorld(true);
}

function initializeInteractiveObjectComponents() {
  const definitions = Array.isArray(manifest?.interactiveObjects) ? manifest.interactiveObjects : [];
  for (const definition of definitions) {
    if (!interactiveObjects.has(definition.id)) createInteractiveObjectComponent(definition);
  }
  applyInteractiveObjectHierarchy();
  return Array.from(interactiveObjects.values());
}

async function loadInteractiveObject(definition) {
  const component = interactiveObjects.get(definition.id) || createInteractiveObjectComponent(definition);
  if (definition.collision?.mode === "unified-glb") {
    return loadUnifiedInteractiveObject(component);
  }
  const { asset, bytes } = await getChunkedAssetBytes(definition.visual, definition.id);
  let visualObject;
  if (asset.renderer === "three-points" || asset.format === "rgb-point-cloud-ply") {
    const geometry = new PLYLoader().parse(collisionArrayBuffer(bytes));
    const positions = geometry.getAttribute("position");
    const colors = geometry.getAttribute("color");
    if (!positions?.count || !colors?.count || positions.count !== colors.count) {
      geometry.dispose();
      throw new Error(`${definition.id}: RGB point visual lacks matching position/color attributes`);
    }
    geometry.computeBoundingBox();
    const material = new THREE.PointsMaterial({
      size: Math.max(0.005, Number(asset.pointSize) || 0.03),
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      opacity: 1,
      depthWrite: true,
    });
    visualObject = new THREE.Points(geometry, material);
    visualObject.name = `${definition.label} accepted RGB point-cloud visual`;
    visualObject.position.copy(vectorFromArray(definition.placement?.generatedCenter)).multiplyScalar(-1);
    visualObject.raycast = () => {};
    component.visualKind = "rgb-points";
    component.splatLocalBounds = geometry.boundingBox?.clone() || component.splatLocalBounds;
  } else {
    visualObject = new SplatMesh({
      fileBytes: bytes,
      fileName: asset.fileName,
      fileType: asset.fileType,
      lod: false,
      raycastable: false,
      onProgress: (event) => {
        if (!event.total) return;
        const pct = Math.round((event.loaded / event.total) * 100);
        modeChip.textContent = `${definition.label} Spark decode ${pct}%`;
      },
    });
    visualObject.name = `${definition.label} independent Gaussian visual`;
    visualObject.position.copy(vectorFromArray(definition.placement?.generatedCenter)).multiplyScalar(-1);
    visualObject.raycast = () => {};
    component.visualKind = "gaussian-splat";
  }
  visualObject.visible = state.showVisual;
  component.content.add(visualObject);
  component.splat = visualObject;
  if (component.visualKind === "gaussian-splat") await visualObject.initialized;
  component.ready = true;
  component.visualReady = true;
  state.interactiveObjectReadyCount += 1;
  if (component.visualKind === "rgb-points") {
    state.interactiveObjectRgbPointCount += Number(asset.vertexCount) || 0;
  } else {
    state.interactiveObjectGaussianCount += Number(asset.vertexCount) || 0;
  }
  state.interactiveObjectVisualPrimitiveCount = state.interactiveObjectGaussianCount
    + state.interactiveObjectRgbPointCount
    + state.interactiveObjectMeshVertexCount;
  state.visualCount = state.staticVisualCount + state.interactiveObjectVisualPrimitiveCount;
  updateHud();
  return component;
}

function collisionArrayBuffer(bytes) {
  return bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? bytes.buffer
    : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function parseObjectCollision(bytes, asset) {
  const fileType = String(asset.fileType || asset.format || "").toLowerCase();
  if (fileType === "glb" || fileType.includes("gltf")) {
    const loader = new GLTFLoader();
    return new Promise((resolve, reject) => {
      loader.parse(collisionArrayBuffer(bytes), "", (result) => resolve(result.scene), reject);
    });
  }
  if (fileType === "obj" || fileType.includes("wavefront")) {
    return new OBJLoader().parse(new TextDecoder().decode(bytes));
  }
  throw new Error(`Unsupported object collider format: ${asset.fileType || asset.format || "unknown"}`);
}

function objectCollisionPlacementMatrix(definition, collisionRoot) {
  const configured = definition.collision?.objectLocalMatrix || definition.collision?.placement?.matrix;
  if (Array.isArray(configured) && configured.length === 16 && configured.every((value) => Number.isFinite(Number(value)))) {
    return new THREE.Matrix4().fromArray(configured.map(Number));
  }
  collisionRoot.updateMatrixWorld(true);
  const sourceBounds = new THREE.Box3().setFromObject(collisionRoot);
  if (sourceBounds.isEmpty()) return new THREE.Matrix4().identity();
  const sourceCenter = sourceBounds.getCenter(new THREE.Vector3());
  const sourceSize = sourceBounds.getSize(new THREE.Vector3());
  const targetSize = vectorFromArray(definition.colliderProxy?.dimensions, [1, 1, 1]);
  const targetCenter = vectorFromArray(definition.colliderProxy?.center);
  const scale = new THREE.Vector3(
    targetSize.x / Math.max(sourceSize.x, 1e-6),
    targetSize.y / Math.max(sourceSize.y, 1e-6),
    targetSize.z / Math.max(sourceSize.z, 1e-6)
  );
  return new THREE.Matrix4()
    .makeTranslation(targetCenter.x, targetCenter.y, targetCenter.z)
    .multiply(new THREE.Matrix4().makeScale(scale.x, scale.y, scale.z))
    .multiply(new THREE.Matrix4().makeTranslation(-sourceCenter.x, -sourceCenter.y, -sourceCenter.z));
}

function registerCollisionMesh(mesh, component, mode) {
  const geometry = mesh.geometry;
  const positions = geometry?.getAttribute?.("position");
  if (!positions?.count) return 0;
  if (!geometry.getAttribute("normal")) geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  if (!geometry.boundsTree) {
    geometry.boundsTree = new MeshBVH(geometry, {
      strategy: SAH,
      indirect: true,
      maxLeafTris: 12,
    });
  }
  mesh.material = new THREE.MeshBasicMaterial({
    color: 0xefb35f,
    transparent: true,
    opacity: state.objectColliderDebugVisible ? 0.18 : 0,
    colorWrite: state.objectColliderDebugVisible,
    depthWrite: false,
    wireframe: true,
    side: THREE.DoubleSide,
  });
  mesh.raycast = acceleratedRaycast;
  mesh.frustumCulled = false;
  mesh.userData.interactiveObjectId = component.definition.id;
  mesh.userData.colliderId = component.definition.id;
  mesh.userData.colliderKind = "object";
  mesh.userData.colliderMode = mode;
  mesh.userData.collisionGateStatus = component.definition.collision?.gate?.status || "missing";
  mesh.userData.colliderLabel = `${component.definition.label} ${mode} collider`;
  mesh.userData.surfaceType = `interactive-object-${mode}-collider`;
  const gatePassed = component.definition.collision?.gate?.status === "passed";
  mesh.userData.walkable = gatePassed && component.definition.collision?.walkable === true;
  mesh.userData.characterCollision = gatePassed
    && component.definition.collision?.characterCollision === true;
  component.collisionMeshes.push(mesh);
  objectCollisionMeshes.push(mesh);
  const index = geometry.index;
  return index ? Math.floor(index.count / 3) : Math.floor(positions.count / 3);
}

function interactiveObjectCollisionBounds(component) {
  if (!component) return null;
  if (component.collisionMode === "none") return null;
  component.group.updateMatrixWorld(true);
  if (["glb", "obj", "unified-glb"].includes(component.collisionMode) && component.collisionRoot) {
    return new THREE.Box3().setFromObject(component.collisionRoot);
  }
  if (!component.proxy) return null;
  const localBounds = component.proxy.geometry.boundingBox
    || (component.proxy.geometry.computeBoundingBox(), component.proxy.geometry.boundingBox);
  return localBounds?.clone().applyMatrix4(component.proxy.matrixWorld) || null;
}

function registerDegradedProxyCollision(component, reason = "missing-glb") {
  const proxy = component.proxy;
  if (!proxy) throw new Error(`${component.definition.id}: no degraded collision proxy is declared`);
  if (!objectCollisionMeshes.includes(proxy)) objectCollisionMeshes.push(proxy);
  proxy.userData.colliderId = component.definition.id;
  proxy.userData.colliderKind = "object";
  proxy.userData.colliderMode = "degraded-box";
  proxy.userData.collisionGateStatus = component.definition.collision?.gate?.status || "missing";
  proxy.userData.colliderLabel = `${component.definition.label} degraded box collider`;
  const explicitlyApprovedProxy = component.definition.collision?.mode === "degraded-box"
    && component.definition.collision?.gate?.status === "passed";
  proxy.userData.walkable = explicitlyApprovedProxy
    && component.definition.collision?.walkable === true;
  proxy.userData.characterCollision = explicitlyApprovedProxy
    && component.definition.collision?.characterCollision === true;
  component.colliderReady = true;
  component.collisionSettled = true;
  component.collisionMode = "degraded-box";
  component.collisionMeshes = [proxy];
  component.collisionFaces = 12;
  component.collisionBounds = interactiveObjectCollisionBounds(component);
  component.collisionError = reason;
  state.objectColliderReadyCount += 1;
  state.objectColliderMeshCount += 1;
  state.objectColliderFaces += 12;
  state.objectColliderDegradedCount += 1;
  return component;
}

async function loadInteractiveObjectCollision(component) {
  const definition = component.definition;
  if (definition.collision?.mode === "unified-glb") {
    return loadUnifiedInteractiveObject(component);
  }
  if (definition.collision?.mode === "none") {
    component.colliderReady = false;
    component.collisionSettled = true;
    component.collisionMode = "none";
    component.collisionMeshes = [];
    component.collisionFaces = 0;
    component.collisionBounds = null;
    component.collisionError = null;
    return component;
  }
  const asset = definition.collision?.asset;
  if (!asset) return registerDegradedProxyCollision(component, "manifest has no GLB/OBJ collider");
  const gateStatus = definition.collision?.gate?.status;
  if (gateStatus !== "passed" && !ALLOW_CANDIDATE_COLLIDERS) {
    return registerDegradedProxyCollision(component, `collision gate is ${gateStatus}`);
  }
  try {
    const { bytes } = await getChunkedAssetBytes(asset, `${definition.id}:collision`);
    const loaded = await parseObjectCollision(bytes, asset);
    const collisionContent = new THREE.Group();
    collisionContent.name = `${definition.id} object collision placement`;
    collisionContent.matrixAutoUpdate = false;
    collisionContent.matrix.copy(objectCollisionPlacementMatrix(definition, loaded));
    collisionContent.add(loaded);
    component.group.add(collisionContent);
    component.collisionRoot = collisionContent;
    let faces = 0;
    const collisionMode = String(asset.fileType || asset.format || "").toLowerCase().includes("obj") ? "obj" : "glb";
    loaded.traverse((node) => {
      if (node.isMesh && !node.isSkinnedMesh) faces += registerCollisionMesh(node, component, collisionMode);
    });
    if (!component.collisionMeshes.length) throw new Error("Collider contains no triangle meshes");
    component.group.updateMatrixWorld(true);
    component.colliderReady = true;
    component.collisionSettled = true;
    component.collisionMode = collisionMode;
    component.collisionFaces = faces;
    component.collisionBounds = interactiveObjectCollisionBounds(component);
    state.objectColliderReadyCount += 1;
    state.objectColliderMeshCount += component.collisionMeshes.length;
    state.objectColliderFaces += faces;
    return component;
  } catch (error) {
    for (const mesh of component.collisionMeshes) {
      const index = objectCollisionMeshes.indexOf(mesh);
      if (index >= 0) objectCollisionMeshes.splice(index, 1);
    }
    component.collisionMeshes = [];
    if (component.collisionRoot) component.group.remove(component.collisionRoot);
    component.collisionRoot = null;
    state.objectColliderLoadErrors.push({ id: definition.id, error: error?.message || String(error) });
    return registerDegradedProxyCollision(component, error?.message || String(error));
  }
}

function setObjectColliderDebugVisible(visible, objectId = null) {
  state.objectColliderDebugVisible = Boolean(visible);
  state.objectColliderDebugObjectId = state.objectColliderDebugVisible && objectId
    ? String(objectId)
    : null;
  for (const mesh of objectCollisionMeshes) {
    if (
      !mesh.material
      || mesh.userData.colliderMode === "degraded-box"
      || mesh.userData.colliderMode === "unified-glb"
    ) continue;
    const selected = !state.objectColliderDebugObjectId
      || mesh.userData.interactiveObjectId === state.objectColliderDebugObjectId;
    const shown = state.objectColliderDebugVisible && selected;
    mesh.material.colorWrite = shown;
    mesh.material.opacity = shown ? 0.18 : 0;
    mesh.material.needsUpdate = true;
  }
  updateHud();
  return state.objectColliderDebugVisible;
}

function ensureInteractiveObjectCollidersLoaded() {
  const components = initializeInteractiveObjectComponents();
  if (!components.length || components.every((component) => component.collisionSettled)) {
    return Promise.resolve(components);
  }
  if (interactiveObjectColliderLoadPromise) return interactiveObjectColliderLoadPromise;
  interactiveObjectColliderLoadPromise = (async () => {
    for (const component of components) {
      if (!component.collisionSettled) await loadInteractiveObjectCollision(component);
    }
    applyEffectiveVisualTransform();
    updateHud();
    return components;
  })().finally(() => {
    interactiveObjectColliderLoadPromise = null;
  });
  return interactiveObjectColliderLoadPromise;
}

function ensureInteractiveObjectsLoaded() {
  const definitions = Array.isArray(manifest?.interactiveObjects) ? manifest.interactiveObjects : [];
  if (!definitions.length || state.interactiveObjectReadyCount === definitions.length) {
    return Promise.resolve(Array.from(interactiveObjects.values()));
  }
  if (interactiveObjectLoadPromise) return interactiveObjectLoadPromise;
  interactiveObjectLoadPromise = (async () => {
    initializeInteractiveObjectComponents();
    for (const definition of definitions) {
      if (interactiveObjects.get(definition.id)?.ready) continue;
      await loadInteractiveObject(definition);
    }
    applyEffectiveVisualTransform();
    setLayerVisibility();
    showToast(`${definitions.length} interactive visual objects ready.`);
    return Array.from(interactiveObjects.values());
  })().finally(() => {
    interactiveObjectLoadPromise = null;
  });
  return interactiveObjectLoadPromise;
}

function ensureVisualSplatLoaded() {
  const objectsReady = state.interactiveObjectReadyCount === state.interactiveObjectCount;
  if (state.visualReady && objectsReady) return Promise.resolve(visualSplat);
  if (visualLoadPromise) return visualLoadPromise;
  state.visualLoadSkipped = false;
  visualLoadPromise = (async () => {
    if (!state.visualReady) await loadVisualSplat(buildEffectiveVisualTransform());
    await ensureInteractiveObjectsLoaded();
    return visualSplat;
  })()
    .then(() => visualSplat)
    .finally(() => {
      visualLoadPromise = null;
    });
  return visualLoadPromise;
}

function parseSemanticMeshPly(text) {
  const lines = text.split(/\r?\n/);
  let i = 0;
  let vertexCount = 0;
  let faceCount = 0;
  for (; i < lines.length; i += 1) {
    const line = lines[i].trim();
    if (line.startsWith("element vertex ")) vertexCount = Number(line.split(/\s+/)[2]);
    if (line.startsWith("element face ")) faceCount = Number(line.split(/\s+/)[2]);
    if (line === "end_header") {
      i += 1;
      break;
    }
  }
  if (!vertexCount || !faceCount) throw new Error("Semantic mesh PLY header missing vertex or face count.");

  const positions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const objectIds = new Int32Array(vertexCount);
  const probabilities = new Float32Array(vertexCount);
  const sourceFaces = new Int32Array(vertexCount);
  const rawColors = new Float32Array(vertexCount * 3);

  for (let v = 0; v < vertexCount; v += 1, i += 1) {
    const parts = lines[i].trim().split(/\s+/);
    positions[v * 3] = Number(parts[0]);
    positions[v * 3 + 1] = Number(parts[1]);
    positions[v * 3 + 2] = Number(parts[2]);
    rawColors[v * 3] = Number(parts[3]) / 255;
    rawColors[v * 3 + 1] = Number(parts[4]) / 255;
    rawColors[v * 3 + 2] = Number(parts[5]) / 255;
    objectIds[v] = Number(parts[6]);
    probabilities[v] = Number(parts[7]);
    sourceFaces[v] = Number(parts[8]);
    const c = semanticColor(objectIds[v]);
    colors[v * 3] = c.r;
    colors[v * 3 + 1] = c.g;
    colors[v * 3 + 2] = c.b;
  }

  const indices = new Uint32Array(faceCount * 3);
  const faceSemantics = new Array(faceCount);
  for (let f = 0; f < faceCount; f += 1, i += 1) {
    const parts = lines[i].trim().split(/\s+/).map(Number);
    if (parts[0] !== 3) throw new Error(`Only triangular faces are supported; got ${parts[0]} at face ${f}`);
    const a = parts[1];
    const b = parts[2];
    const c = parts[3];
    indices[f * 3] = a;
    indices[f * 3 + 1] = b;
    indices[f * 3 + 2] = c;
    const objectId = Math.round((objectIds[a] + objectIds[b] + objectIds[c]) / 3);
    const probability = (probabilities[a] + probabilities[b] + probabilities[c]) / 3;
    const sourceFace = Math.round((sourceFaces[a] + sourceFaces[b] + sourceFaces[c]) / 3);
    faceSemantics[f] = { objectId, probability, sourceFace };
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute("rawColor", new THREE.BufferAttribute(rawColors, 3));
  geometry.setAttribute("objectId", new THREE.Int32BufferAttribute(objectIds, 1));
  geometry.setAttribute("objectProbability", new THREE.BufferAttribute(probabilities, 1));
  geometry.setAttribute("sourceFace", new THREE.Int32BufferAttribute(sourceFaces, 1));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  return { geometry, faceSemantics, vertexCount, faceCount, hasSemantics: true };
}

function parseBinaryMeshPly(bytes) {
  const arrayBuffer = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? bytes.buffer
    : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const geometry = new PLYLoader().parse(arrayBuffer);
  const position = geometry.getAttribute("position");
  if (!position?.count) throw new Error("Binary mesh PLY is missing vertex positions.");
  if (!geometry.index?.count || geometry.index.count % 3 !== 0) {
    throw new Error("Binary mesh PLY is missing triangular face indices.");
  }
  if (!geometry.getAttribute("color")) {
    const colors = new Float32Array(position.count * 3);
    colors.fill(0.72);
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  }
  if (!geometry.getAttribute("normal")) geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  return {
    geometry,
    faceSemantics: [],
    vertexCount: position.count,
    faceCount: geometry.index.count / 3,
    hasSemantics: false,
  };
}

function parseColliderMeshPly(bytes, asset) {
  if (asset.format === "semantic-ascii-mesh-ply") {
    return parseSemanticMeshPly(new TextDecoder().decode(bytes));
  }
  if (asset.format === "open3d-binary-little-endian-triangle-mesh-ply") {
    return parseBinaryMeshPly(bytes);
  }
  throw new Error(`Unsupported collider PLY format: ${asset.format || "unknown"}`);
}

function estimateMainWalkableFloorY(geometry) {
  const index = geometry.index;
  const position = geometry.getAttribute("position");
  const faceCount = index ? Math.floor(index.count / 3) : Math.floor(position.count / 3);
  const bounds = geometry.boundingBox;
  const upSign = Math.sign(ROBOT_UP.y) || 1;
  const minHeight = Math.min(bounds.min.y * upSign, bounds.max.y * upSign);
  const heightLimit = minHeight + boxSize(bounds).y * 0.45;
  const bins = new Map();
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const normal = new THREE.Vector3();

  const readVertex = (vertexIndex, target) => {
    target.set(position.getX(vertexIndex), position.getY(vertexIndex), position.getZ(vertexIndex));
  };

  for (let face = 0; face < faceCount; face += 1) {
    const base = face * 3;
    const ia = index ? index.getX(base) : base;
    const ib = index ? index.getX(base + 1) : base + 1;
    const ic = index ? index.getX(base + 2) : base + 2;
    readVertex(ia, a);
    readVertex(ib, b);
    readVertex(ic, c);
    ab.subVectors(b, a);
    ac.subVectors(c, a);
    normal.crossVectors(ab, ac);
    const area2 = normal.length();
    if (area2 < 1e-7) continue;
    normal.divideScalar(area2);
    if (normal.dot(ROBOT_UP) < 0.55) continue;
    const y = (a.y + b.y + c.y) / 3;
    const height = y * upSign;
    if (height > heightLimit) continue;
    const bin = Math.round(height * 2) / 2;
    bins.set(bin, (bins.get(bin) || 0) + area2 * 0.5);
  }

  let bestY = null;
  let bestArea = 0;
  for (const [y, area] of bins) {
    if (area > bestArea) {
      bestY = y;
      bestArea = area;
    }
  }
  return Number.isFinite(bestY) ? bestY / upSign : null;
}

async function loadColliderMesh() {
  const assetKey = manifest?.collisionWorld?.sceneAssetKey || "collider";
  const { asset, bytes } = await getChunkedBytes(assetKey);
  state.colliderAssetId = asset.id;
  state.colliderUrl = asset.fileName;
  state.colliderVertices = asset.vertexCount || 0;
  state.colliderFaces = asset.faceCount || 0;
  state.colliderSha256 = asset.sha256 || "";
  state.colliderSourcePath = asset.sourcePath || "";
  state.colliderFormat = asset.format || "ply";

  const parsed = parseColliderMeshPly(bytes, asset);
  if (asset.vertexCount && parsed.vertexCount !== asset.vertexCount) {
    throw new Error(`Collider vertex count mismatch: expected ${asset.vertexCount}, got ${parsed.vertexCount}`);
  }
  if (asset.faceCount && parsed.faceCount !== asset.faceCount) {
    throw new Error(`Collider face count mismatch: expected ${asset.faceCount}, got ${parsed.faceCount}`);
  }
  state.colliderHasSemantics = parsed.hasSemantics;
  if (!parsed.hasSemantics) state.semanticColor = false;
  const rawColliderBounds = parsed.geometry.boundingBox.clone();
  if (!scenePresentationPivot) scenePresentationPivot = boxCenter(colliderMetaBox || rawColliderBounds);
  state.scenePresentationPivot = roundedVector(scenePresentationPivot);
  applyScenePresentationToGeometry(parsed.geometry, scenePresentationPivot);
  colliderFaceSemantics = parsed.faceSemantics || [];
  colliderBounds = parsed.geometry.boundingBox.clone();
  robotFloorYEstimate = estimateMainWalkableFloorY(parsed.geometry);
  state.robotFloorYEstimate = Number.isFinite(robotFloorYEstimate)
    ? Number(robotFloorYEstimate.toFixed(3))
    : null;
  parsed.geometry.boundsTree = new MeshBVH(parsed.geometry, {
    strategy: SAH,
    indirect: true,
    maxLeafTris: 12,
  });
  state.colliderAcceleration = "three-mesh-bvh 0.8.3";

  const material = new THREE.MeshStandardMaterial({
    vertexColors: parsed.geometry.hasAttribute("color"),
    side: THREE.DoubleSide,
    roughness: 0.86,
    metalness: 0.02,
    transparent: true,
    opacity: 0.2,
    depthWrite: false,
  });
  colliderMesh = new THREE.Mesh(parsed.geometry, material);
  colliderMesh.name = asset.label || "PLY mesh collider proxy";
  colliderMesh.userData.colliderLabel = asset.label || asset.id || "mesh collider";
  colliderMesh.userData.surfaceType = parsed.hasSemantics ? "semantic-mesh-ply-collider" : "tsdf-mesh-ply-collider";
  colliderMesh.userData.walkable = true;
  colliderMesh.userData.characterCollision = true;
  colliderMesh.userData.cameraCollision = true;
  colliderMesh.raycast = acceleratedRaycast;
  colliderLayer.add(colliderMesh);

  colliderWire = new THREE.LineSegments(
    createSampledWireGeometry(parsed.geometry),
    new THREE.LineBasicMaterial({
      color: 0xefb35f,
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
    })
  );
  colliderWire.name = "PLY collider sampled wire overlay";
  colliderWire.raycast = () => {};
  colliderLayer.add(colliderWire);

  state.colliderVertices = parsed.vertexCount;
  state.colliderFaces = parsed.faceCount;
  state.colliderReady = true;
  showToast(parsed.hasSemantics ? "Semantic mesh PLY collider loaded." : "TSDF mesh PLY collider loaded.");
  updateHud();
  return colliderBounds;
}

function collisionTargets({ walkable = false, characterCollision = false } = {}) {
  const targets = [];
  if (colliderMesh) targets.push(colliderMesh);
  for (const mesh of objectCollisionMeshes) {
    if (walkable && mesh.userData.walkable !== true) continue;
    if (characterCollision && mesh.userData.characterCollision !== true) continue;
    targets.push(mesh);
  }
  return targets;
}

function collisionMeshPairIntersects(left, right) {
  left.updateMatrixWorld(true);
  right.updateMatrixWorld(true);

  const leftBounds = new THREE.Box3().setFromObject(left);
  const rightBounds = new THREE.Box3().setFromObject(right);
  if (leftBounds.isEmpty() || rightBounds.isEmpty() || !leftBounds.intersectsBox(rightBounds)) {
    return false;
  }

  for (const mesh of [left, right]) {
    if (!mesh.geometry?.boundsTree) {
      mesh.geometry.boundsTree = new MeshBVH(mesh.geometry, {
        strategy: SAH,
        indirect: true,
        maxLeafTris: 12,
      });
    }
  }

  const rightToLeft = new THREE.Matrix4()
    .copy(left.matrixWorld)
    .invert()
    .multiply(right.matrixWorld);
  return left.geometry.boundsTree.intersectsGeometry(right.geometry, rightToLeft);
}

function inspectSceneInterpenetrations() {
  const targets = collisionTargets();
  const intersectionsByPair = new Map();
  let meshPairTests = 0;

  for (let leftIndex = 0; leftIndex < targets.length; leftIndex += 1) {
    const left = targets[leftIndex];
    const leftId = String(left.userData.colliderId || left.userData.interactiveObjectId || "scene");
    for (let rightIndex = leftIndex + 1; rightIndex < targets.length; rightIndex += 1) {
      const right = targets[rightIndex];
      const rightId = String(right.userData.colliderId || right.userData.interactiveObjectId || "scene");
      if (leftId === rightId) continue;
      meshPairTests += 1;
      if (!collisionMeshPairIntersects(left, right)) continue;

      const pairIds = [leftId, rightId].sort();
      const pairKey = pairIds.join("::");
      if (intersectionsByPair.has(pairKey)) continue;
      intersectionsByPair.set(pairKey, {
        colliderIds: pairIds,
        objectIds: pairIds.filter((id) => id !== "scene"),
        colliderKinds: [
          left.userData.colliderKind || (leftId === "scene" ? "scene" : "object"),
          right.userData.colliderKind || (rightId === "scene" ? "scene" : "object"),
        ],
        leftBounds: serializeBox(new THREE.Box3().setFromObject(left)),
        rightBounds: serializeBox(new THREE.Box3().setFromObject(right)),
      });
    }
  }

  const intersections = Array.from(intersectionsByPair.values());
  return {
    status: intersections.length ? "failed" : "passed",
    targetCount: targets.length,
    meshPairTests,
    intersectionCount: intersections.length,
    intersections,
    limitation: "Conservative triangle-surface intersection gate; accepted support contact still requires six-view placement review.",
  };
}

function intersectCollisionWorld(activeRaycaster, options = {}) {
  const targets = collisionTargets(options);
  if (!targets.length) return [];
  return activeRaycaster.intersectObjects(targets, false);
}

function worldHitNormal(hit) {
  const normal = hit.face?.normal.clone() || ROBOT_UP.clone();
  return normal.applyMatrix3(new THREE.Matrix3().getNormalMatrix(hit.object.matrixWorld)).normalize();
}

function applyColliderMode() {
  if (colliderMesh) {
    colliderMesh.visible = state.colliderRenderMode !== "hidden";
    colliderMesh.material.opacity = state.colliderRenderMode === "solid" ? 0.32 : 0.055;
  }
  if (colliderWire) {
    colliderWire.visible = state.colliderRenderMode === "wire";
  }
  updateHud();
}

function applySemanticColorMode() {
  if (!colliderMesh) return;
  if (!state.colliderHasSemantics) {
    state.semanticColor = false;
    return;
  }
  const geometry = colliderMesh.geometry;
  const color = geometry.getAttribute("color");
  const rawColor = geometry.getAttribute("rawColor");
  if (!color || !rawColor) return;
  if (state.semanticColor) {
    const objectId = geometry.getAttribute("objectId");
    for (let i = 0; i < color.count; i += 1) {
      const c = semanticColor(objectId.getX(i));
      color.setXYZ(i, c.r, c.g, c.b);
    }
  } else {
    for (let i = 0; i < color.count; i += 1) {
      color.setXYZ(i, rawColor.getX(i), rawColor.getY(i), rawColor.getZ(i));
    }
  }
  color.needsUpdate = true;
}

function createRobot() {
  const group = new THREE.Group();
  group.name = "Video2Mesh mesh-proxy robot";

  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: 0x58d7c9,
    roughness: 0.48,
    metalness: 0.16,
    emissive: 0x092e2a,
    emissiveIntensity: 0.42,
  });
  const accentMaterial = new THREE.MeshStandardMaterial({
    color: 0xefb35f,
    roughness: 0.58,
    metalness: 0.08,
    emissive: 0x2b1804,
    emissiveIntensity: 0.35,
  });
  const darkMaterial = new THREE.MeshStandardMaterial({
    color: 0x11191c,
    roughness: 0.62,
    metalness: 0.05,
  });
  const eyeMaterial = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    emissive: 0x58d7c9,
    emissiveIntensity: 1.2,
    roughness: 0.2,
  });

  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.32, 0.58, 6, 14), bodyMaterial);
  body.position.y = 0.78;
  group.add(body);

  const head = new THREE.Mesh(new THREE.SphereGeometry(0.28, 20, 16), accentMaterial);
  head.scale.set(1.08, 0.88, 1.08);
  head.position.y = 1.34;
  group.add(head);

  const visor = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.08, 0.045), eyeMaterial);
  visor.position.set(0, 1.38, 0.255);
  group.add(visor);

  const antenna = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.38, 8), accentMaterial);
  antenna.position.set(0.12, 1.68, 0);
  antenna.rotation.z = -0.24;
  group.add(antenna);

  const antennaTip = new THREE.Mesh(new THREE.SphereGeometry(0.055, 12, 8), eyeMaterial);
  antennaTip.position.set(0.165, 1.87, 0);
  group.add(antennaTip);

  const leftArm = new THREE.Mesh(new THREE.CapsuleGeometry(0.075, 0.44, 4, 10), darkMaterial);
  leftArm.position.set(-0.42, 0.86, 0.02);
  leftArm.rotation.z = 0.2;
  group.add(leftArm);

  const rightArm = leftArm.clone();
  rightArm.position.x = 0.42;
  rightArm.rotation.z = -0.2;
  group.add(rightArm);

  const footGeometry = new THREE.BoxGeometry(0.28, 0.12, 0.34);
  const leftFoot = new THREE.Mesh(footGeometry, darkMaterial);
  leftFoot.position.set(-0.17, 0.08, 0.05);
  group.add(leftFoot);

  const rightFoot = leftFoot.clone();
  rightFoot.position.x = 0.17;
  group.add(rightFoot);

  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.52, 0.012, 8, 40),
    new THREE.MeshBasicMaterial({ color: 0x58d7c9, transparent: true, opacity: 0.52 })
  );
  ring.name = "robot ground probe ring";
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.035;
  group.add(ring);

  group.traverse((child) => {
    if (child.isMesh) child.raycast = () => {};
  });
  group.userData.height = 1.72;
  group.userData.groundOffset = 0.02;
  group.userData.upSign = Math.sign(ROBOT_UP.y) || 1;
  group.scale.y = group.userData.upSign;
  group.visible = state.robotEnabled;
  return group;
}

function getColliderProbeHeights() {
  if (!colliderBounds) return { originY: 40, directionY: -1, range: 120 };
  const size = boxSize(colliderBounds);
  const upSign = Math.sign(ROBOT_UP.y) || 1;
  return {
    originY: upSign > 0
      ? colliderBounds.max.y + Math.max(4, size.y * 0.25)
      : colliderBounds.min.y - Math.max(4, size.y * 0.25),
    directionY: -upSign,
    range: Math.max(16, size.y + 12),
  };
}

function groundProbe(position, { preferFloor = true, maxAbove = Infinity } = {}) {
  if (!collisionTargets({ walkable: true }).length) return null;
  const { originY, directionY, range } = getColliderProbeHeights();
  groundRaycaster.set(
    new THREE.Vector3(position.x, originY, position.z),
    new THREE.Vector3(0, directionY, 0)
  );
  groundRaycaster.near = 0;
  groundRaycaster.far = range;
  const hits = intersectCollisionWorld(groundRaycaster, { walkable: true });
  if (!hits.length) return null;
  const walkableHits = hits.filter((hit) => {
    const normal = worldHitNormal(hit);
    if (normal.dot(ROBOT_UP) <= 0.3) return false;
    const heightAboveProbe = hit.point.clone().sub(position).dot(ROBOT_UP);
    return heightAboveProbe <= maxAbove;
  });
  if (!walkableHits.length) return null;
  if (preferFloor && Number.isFinite(robotFloorYEstimate)) {
    const floorHit = walkableHits
      .filter((hit) => Math.abs(hit.point.y - robotFloorYEstimate) <= 2.4)
      .sort((a, b) => Math.abs(a.point.y - robotFloorYEstimate) - Math.abs(b.point.y - robotFloorYEstimate))[0];
    if (floorHit) return floorHit;
  }
  return walkableHits[0];
}

function isRobotStepBlocked(fromPosition, toPosition) {
  if (!collisionTargets({ characterCollision: true }).length || !state.robotObstacleCollision) return false;
  const direction = toPosition.clone().sub(fromPosition);
  direction.y = 0;
  const distance = direction.length();
  if (distance < 1e-4) return false;
  direction.normalize();
  const origin = fromPosition.clone();
  origin.addScaledVector(ROBOT_UP, 0.58);
  obstacleRaycaster.set(origin, direction);
  obstacleRaycaster.near = 0.04;
  obstacleRaycaster.far = Math.min(0.72, distance + 0.24);
  const hits = intersectCollisionWorld(obstacleRaycaster, { characterCollision: true });
  const blockingHit = hits.find((hit) => Math.abs(worldHitNormal(hit).dot(ROBOT_UP)) < 0.52);
  if (!blockingHit) return false;
  state.lastCollisionKind = blockingHit.object.userData.colliderKind || "scene";
  state.lastCollisionObjectId = blockingHit.object.userData.interactiveObjectId || null;
  return true;
}

function isInsideColliderFootprint(position, margin = 0.8) {
  if (!colliderBounds) return false;
  return (
    position.x >= colliderBounds.min.x - margin &&
    position.x <= colliderBounds.max.x + margin &&
    position.z >= colliderBounds.min.z - margin &&
    position.z <= colliderBounds.max.z + margin
  );
}

function findGroundForStep(fromPosition, move, distance) {
  const direction = move.clone().normalize();
  const right = new THREE.Vector3().crossVectors(direction, ROBOT_UP).normalize();
  const distances = [distance, distance * 0.68, distance * 0.42, distance * 0.24];
  const lateralOffsets = [0, 0.28, -0.28, 0.58, -0.58, 0.92, -0.92];
  const hits = [];

  for (const travel of distances) {
    const base = fromPosition.clone().addScaledVector(direction, travel);
    for (const lateral of lateralOffsets) {
      const candidate = base.clone().addScaledVector(right, lateral);
      const hit = groundProbe(candidate, { preferFloor: true });
      if (!hit) continue;
      const intended = fromPosition.clone().addScaledVector(direction, distance);
      const lateralScore = Math.hypot(hit.point.x - intended.x, hit.point.z - intended.z);
      const yScore = Number.isFinite(robotFloorYEstimate)
        ? Math.abs(hit.point.y - robotFloorYEstimate) * 0.2
        : 0;
      hits.push({ hit, score: lateralScore + yScore });
    }
  }

  hits.sort((a, b) => a.score - b.score);
  if (hits.length) return { point: hits[0].hit.point.clone(), source: "mesh-nearby" };

  const fallback = fromPosition.clone().addScaledVector(direction, distance);
  if (!state.robotObstacleCollision && Number.isFinite(robotFloorYEstimate) && isInsideColliderFootprint(fallback)) {
    fallback.y = robotFloorYEstimate;
    return { point: fallback, source: "floor-plane-fallback" };
  }
  return null;
}

function respawnRobotAt(point, source = "manual", { yaw = null } = {}) {
  if (!colliderMesh) return false;
  if (!robotGroup) {
    robotGroup = createRobot();
    robotLayer.add(robotGroup);
  }
  const spawn = point.clone();
  spawn.addScaledVector(ROBOT_UP, robotGroup.userData.groundOffset);
  robotGroup.position.copy(spawn);
  robotGroup.rotation.y = Number.isFinite(yaw)
    ? yaw
    : Math.atan2(camera.position.x - spawn.x, camera.position.z - spawn.z);
  robotGroundY = spawn.y;
  robotVerticalSpeed = 0;
  robotAirTime = 0;
  robotJumpStartPosition = null;

  state.robotReady = true;
  state.robotGrounded = true;
  state.robotBlocked = false;
  state.robotVerticalSpeed = 0;
  state.robotJumpPhase = "grounded";
  state.robotAirTime = 0;
  state.robotJumpApex = 0;
  state.robotLastLanding = source;
  state.robotSupportFace = null;
  state.robotSpawnSource = source;
  state.robotPosition = roundedVector(robotGroup.position);
  state.robotYaw = Number(robotGroup.rotation.y.toFixed(4));
  if (state.robotFollowCamera) updateRobotFollowCamera(0, true);
  updateHud();
  return true;
}

function findRobotSpawnHit() {
  if (!colliderMesh || !colliderBounds) return null;
  const center = transformedVisualBounds ? boxCenter(transformedVisualBounds) : boxCenter(colliderBounds);
  const size = boxSize(colliderBounds);
  const zOffsets = [-0.18, -0.28, -0.08, 0.08, 0.2, -0.38].map((value) => value * size.z);
  const xOffsets = [0, -0.08, 0.08, -0.16, 0.16].map((value) => value * size.x);
  const candidates = [];
  for (const z of zOffsets) {
    for (const x of xOffsets) {
      candidates.push(center.clone().add(new THREE.Vector3(x, 0, z)));
    }
  }
  candidates.push(boxCenter(colliderBounds).add(new THREE.Vector3(0, 0, -size.z * 0.28)));
  candidates.push(boxCenter(colliderBounds).add(new THREE.Vector3(size.x * 0.1, 0, -size.z * 0.22)));

  const scoredHits = [];
  for (const candidate of candidates) {
    const hit = groundProbe(candidate, { preferFloor: true });
    if (!hit) continue;
    const yScore = Number.isFinite(robotFloorYEstimate)
      ? Math.abs(hit.point.y - robotFloorYEstimate)
      : 0;
    const centerScore = Math.hypot(hit.point.x - center.x, hit.point.z - center.z) * 0.035;
    scoredHits.push({ hit, score: yScore + centerScore });
  }
  scoredHits.sort((a, b) => a.score - b.score);
  return scoredHits[0]?.hit || null;
}

function fixedRobotSpawnFromManifest() {
  const config = manifest?.initialState?.robotSpawn;
  if (!config || config.coordinateFrame !== "visual_native" || !baseVisualTransform) return null;
  if (!Array.isArray(config.groundPoint) || config.groundPoint.length !== 3) return null;
  const transform = applyScenePresentationToVisualTransform(baseVisualTransform);
  const point = transformVisualPoint(vectorFromArray(config.groundPoint), transform);
  const nativeForward = vectorFromArray(config.forward);
  let yaw = null;
  if (nativeForward.lengthSq() > 1e-8) {
    const forward = transformVisualDirection(nativeForward, transform);
    forward.y = 0;
    if (forward.lengthSq() > 1e-8) yaw = Math.atan2(forward.x, forward.z);
  }
  return {
    point,
    yaw,
    source: config.id || "manifest-fixed",
  };
}

function spawnRobotAtBounds() {
  if (!colliderMesh || !colliderBounds) return;
  if (!robotGroup) {
    robotGroup = createRobot();
    robotLayer.add(robotGroup);
  }

  const fixedSpawn = fixedRobotSpawnFromManifest();
  if (fixedSpawn) {
    respawnRobotAt(fixedSpawn.point, fixedSpawn.source, { yaw: fixedSpawn.yaw });
    return;
  }

  const spawnHit = findRobotSpawnHit();
  if (spawnHit) {
    respawnRobotAt(spawnHit.point, "auto-floor");
    return;
  }
  const fallback = boxCenter(colliderBounds);
  fallback.y = Number.isFinite(robotFloorYEstimate) ? robotFloorYEstimate : fallback.y;
  respawnRobotAt(fallback, "bbox-fallback");
}

function getCameraPlanarBasis() {
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  forward.y = 0;
  if (forward.lengthSq() < 1e-5) forward.set(0, 0, -1);
  forward.normalize();
  const right = new THREE.Vector3().crossVectors(forward, ROBOT_UP).normalize();
  return { forward, right };
}

function getRobotMoveVector() {
  const { forward, right } = getCameraPlanarBasis();
  const move = new THREE.Vector3();
  if (keyState.has("KeyW") || keyState.has("ArrowUp")) move.add(forward);
  if (keyState.has("KeyS") || keyState.has("ArrowDown")) move.sub(forward);
  if (keyState.has("KeyA") || keyState.has("ArrowLeft")) move.sub(right);
  if (keyState.has("KeyD") || keyState.has("ArrowRight")) move.add(right);
  if (move.lengthSq() > 0) move.normalize();
  return move;
}

function moveRobotOnMesh(move, distance) {
  if (!robotGroup || !colliderMesh || move.lengthSq() <= 0) return false;
  const current = robotGroup.position.clone();
  const target = current.clone().addScaledVector(move.clone().normalize(), distance);
  state.robotBlocked = false;

  if (!state.robotObstacleCollision && Number.isFinite(robotFloorYEstimate) && isInsideColliderFootprint(target)) {
    target.y = robotFloorYEstimate + ROBOT_UP.y * robotGroup.userData.groundOffset;
    robotGroup.position.copy(target);
    robotGroundY = robotGroup.position.y;
    state.robotGrounded = true;
    const targetYaw = Math.atan2(move.x, move.z);
    const yawDelta = Math.atan2(
      Math.sin(targetYaw - robotGroup.rotation.y),
      Math.cos(targetYaw - robotGroup.rotation.y)
    );
    robotGroup.rotation.y += yawDelta;
    state.robotPosition = roundedVector(robotGroup.position);
    state.robotYaw = Number(robotGroup.rotation.y.toFixed(4));
    return true;
  }

  if (isRobotStepBlocked(current, target)) {
    state.robotBlocked = true;
    return false;
  }
  const stepGround = findGroundForStep(current, move, distance);
  if (!stepGround) {
    state.robotBlocked = true;
    state.robotGrounded = true;
    return false;
  }
  const next = stepGround.point.clone();
  next.addScaledVector(ROBOT_UP, robotGroup.userData.groundOffset);
  const maxStep = stepGround.source === "floor-plane-fallback" ? 1.25 : 0.92;
  if (robotGroundY != null && Math.abs(next.y - robotGroundY) > maxStep) {
    state.robotBlocked = true;
    return false;
  }
  robotGroup.position.copy(next);
  robotGroundY = robotGroup.position.y;
  state.robotGrounded = true;
  const targetYaw = Math.atan2(move.x, move.z);
  const yawDelta = Math.atan2(
    Math.sin(targetYaw - robotGroup.rotation.y),
    Math.cos(targetYaw - robotGroup.rotation.y)
  );
  robotGroup.rotation.y += yawDelta;
  state.robotPosition = roundedVector(robotGroup.position);
  state.robotYaw = Number(robotGroup.rotation.y.toFixed(4));
  return true;
}

function moveRobotAirborne(move, distance) {
  if (!robotGroup || !colliderMesh || move.lengthSq() <= 0) return false;
  const current = robotGroup.position.clone();
  const target = current.clone().addScaledVector(move.clone().normalize(), distance);
  state.robotBlocked = false;
  if (state.robotObstacleCollision && isRobotStepBlocked(current, target)) {
    state.robotBlocked = true;
    return false;
  }
  if (!isInsideColliderFootprint(target, 1.8)) {
    state.robotBlocked = true;
    return false;
  }
  robotGroup.position.x = target.x;
  robotGroup.position.z = target.z;
  return true;
}

function startRobotJump() {
  if (!robotGroup || !state.robotEnabled || !state.robotReady || !state.robotGrounded) return false;
  robotVerticalSpeed = ROBOT_JUMP_SPEED;
  robotAirTime = 0;
  robotJumpStartPosition = robotGroup.position.clone();
  state.robotGrounded = false;
  state.robotBlocked = false;
  state.robotVerticalSpeed = Number(robotVerticalSpeed.toFixed(3));
  state.robotJumpPhase = "jump";
  state.robotAirTime = 0;
  state.robotJumpApex = 0;
  state.robotLastLanding = "airborne";
  state.robotSupportFace = null;
  showToast("Robot jumped from the mesh surface.");
  updateHud();
  return true;
}

function resolveRobotCeiling(current, next) {
  if (!collisionTargets({ characterCollision: true }).length || robotVerticalSpeed <= 0) return false;
  const rise = next.clone().sub(current).dot(ROBOT_UP);
  if (rise <= 0) return false;
  const height = robotGroup.userData.height;
  const headOrigin = current.clone().addScaledVector(ROBOT_UP, Math.max(0.1, height - 0.08));
  verticalRaycaster.set(headOrigin, ROBOT_UP);
  verticalRaycaster.near = 0.015;
  verticalRaycaster.far = rise + 0.11;
  const hit = intersectCollisionWorld(verticalRaycaster, { characterCollision: true })[0];
  if (!hit) return false;
  next.copy(hit.point).addScaledVector(ROBOT_UP, -(height + 0.035));
  robotVerticalSpeed = Math.min(0, robotVerticalSpeed);
  state.robotJumpPhase = "fall";
  return true;
}

function landRobotOn(hit) {
  const support = hit.point.clone().addScaledVector(ROBOT_UP, robotGroup.userData.groundOffset);
  robotGroup.position.copy(support);
  robotGroundY = support.y;
  robotVerticalSpeed = 0;
  robotAirTime = 0;
  robotJumpStartPosition = null;
  state.robotGrounded = true;
  state.robotBlocked = false;
  state.robotVerticalSpeed = 0;
  state.robotJumpPhase = "grounded";
  state.robotAirTime = 0;
  state.robotLastLanding = "mesh";
  state.robotSupportFace = Number.isInteger(hit.faceIndex) ? hit.faceIndex : null;
  state.robotSupportColliderId = hit.object.userData.colliderId || "scene";
  showToast(`Robot landed on mesh face ${state.robotSupportFace ?? "n/a"}.`);
}

function updateRobotVertical(dt) {
  if (!robotGroup || state.robotGrounded) return;
  const current = robotGroup.position.clone();
  robotAirTime += dt;
  robotVerticalSpeed -= ROBOT_GRAVITY * dt;
  const next = current.clone().addScaledVector(ROBOT_UP, robotVerticalSpeed * dt);

  if (robotVerticalSpeed > 0) {
    resolveRobotCeiling(current, next);
  } else {
    const fallDistance = Math.abs(robotVerticalSpeed * dt);
    const hit = groundProbe(next, {
      preferFloor: false,
      maxAbove: fallDistance + 0.08,
    });
    if (hit) {
      const support = hit.point.clone().addScaledVector(ROBOT_UP, robotGroup.userData.groundOffset);
      const currentHeight = current.clone().sub(support).dot(ROBOT_UP);
      const nextHeight = next.clone().sub(support).dot(ROBOT_UP);
      const crossingTolerance = Math.max(0.035, Math.abs(robotVerticalSpeed * dt) * 0.2);
      if (currentHeight >= -0.055 && nextHeight <= crossingTolerance) {
        landRobotOn(hit);
        return;
      }
    }
  }

  robotGroup.position.copy(next);
  state.robotVerticalSpeed = Number(robotVerticalSpeed.toFixed(3));
  state.robotJumpPhase = robotVerticalSpeed > 0 ? "jump" : "fall";
  state.robotAirTime = Number(robotAirTime.toFixed(3));
  if (robotJumpStartPosition) {
    const height = robotGroup.position.clone().sub(robotJumpStartPosition).dot(ROBOT_UP);
    state.robotJumpApex = Number(Math.max(state.robotJumpApex, height).toFixed(3));
  }

  const floorHeight = Number.isFinite(robotFloorYEstimate)
    ? robotGroup.position.clone().sub(new THREE.Vector3(robotGroup.position.x, robotFloorYEstimate, robotGroup.position.z)).dot(ROBOT_UP)
    : 0;
  if (!isInsideColliderFootprint(robotGroup.position, 3) || floorHeight < -ROBOT_FALL_RESPAWN_DISTANCE) {
    spawnRobotAtBounds();
    showToast("Robot left the collider and returned to its fixed spawn.");
  }
}

function stepRobotByCamera(direction) {
  if (!robotGroup || !state.robotEnabled) return;
  const { forward, right } = getCameraPlanarBasis();
  const move = new THREE.Vector3();
  if (direction === "forward") move.copy(forward);
  if (direction === "back") move.copy(forward).multiplyScalar(-1);
  if (direction === "left") move.copy(right).multiplyScalar(-1);
  if (direction === "right") move.copy(right);
  const moved = state.robotGrounded
    ? moveRobotOnMesh(move, 0.85)
    : moveRobotAirborne(move, 0.5);
  if (state.robotFollowCamera) updateRobotFollowCamera(0, true);
  updateHud();
  showToast(moved ? "Robot stepped on mesh floor." : "Robot step blocked by mesh probe.");
}

function pulseRobotFromKey(code) {
  if (!robotGroup || !state.robotEnabled) return;
  const { forward, right } = getCameraPlanarBasis();
  const move = new THREE.Vector3();
  if (code === "KeyW" || code === "ArrowUp") move.copy(forward);
  if (code === "KeyS" || code === "ArrowDown") move.copy(forward).multiplyScalar(-1);
  if (code === "KeyA" || code === "ArrowLeft") move.copy(right).multiplyScalar(-1);
  if (code === "KeyD" || code === "ArrowRight") move.copy(right);
  if (move.lengthSq() === 0) return;
  if (state.robotGrounded) moveRobotOnMesh(move, 0.32);
  else moveRobotAirborne(move, 0.5);
  if (state.robotFollowCamera) updateRobotFollowCamera(0, true);
  updateHud();
}

function updateRobot(dt) {
  if (!robotGroup || !state.robotEnabled) {
    state.robotSpeed = 0;
    return;
  }

  const move = getRobotMoveVector();
  const isMoving = move.lengthSq() > 0;
  const speed = (keyState.has("ShiftLeft") || keyState.has("ShiftRight")) ? 4.8 : 2.35;
  state.robotSpeed = isMoving ? speed : 0;

  if (isMoving) {
    if (state.robotGrounded) moveRobotOnMesh(move, speed * dt);
    else moveRobotAirborne(move, speed * dt);

    const targetYaw = Math.atan2(move.x, move.z);
    const yawDelta = Math.atan2(
      Math.sin(targetYaw - robotGroup.rotation.y),
      Math.cos(targetYaw - robotGroup.rotation.y)
    );
    robotGroup.rotation.y += yawDelta * Math.min(1, dt * 12);
    if (state.robotGrounded) {
      robotBobPhase += dt * speed * 6.2;
      robotGroup.scale.y = robotGroup.userData.upSign * (1 + Math.sin(robotBobPhase) * 0.025);
    } else {
      robotGroup.scale.y = robotGroup.userData.upSign;
    }
  } else {
    if (state.robotGrounded) {
      robotBobPhase += dt * 1.8;
      robotGroup.scale.y = robotGroup.userData.upSign * (1 + Math.sin(robotBobPhase) * 0.008);
    } else {
      robotGroup.scale.y = robotGroup.userData.upSign;
    }
  }

  updateRobotVertical(dt);

  state.robotPosition = roundedVector(robotGroup.position);
  state.robotYaw = Number(robotGroup.rotation.y.toFixed(4));
  if (state.robotFollowCamera) updateRobotFollowCamera(dt);
}

function updateRobotFollowCamera(dt, immediate = false) {
  if (!robotGroup || state.cameraMode !== "orbit") return;
  camera.up.copy(ROBOT_UP);
  const target = robotGroup.position.clone().addScaledVector(ROBOT_UP, 0.82);
  const yaw = robotGroup.rotation.y;
  const followOffset = new THREE.Vector3(
    Math.sin(yaw + Math.PI) * 4.8,
    0,
    Math.cos(yaw + Math.PI) * 4.8
  );
  const desiredCamera = target.clone().add(followOffset).addScaledVector(ROBOT_UP, 2.15);
  setSmoothCameraPose(desiredCamera, target, { immediate });
  state.cameraPreset = "robotFollow";
  state.cameraPresetSource = "robot-follow";
}

function cameraAnchorFromScene(bounds, { preferRobot = false } = {}) {
  const center = boxCenter(bounds);
  const floorY = Number.isFinite(robotFloorYEstimate)
    ? robotFloorYEstimate
    : (ROBOT_UP.y > 0 ? bounds.min.y : bounds.max.y);
  const anchor = preferRobot && robotGroup ? robotGroup.position.clone() : center.clone();
  anchor.y = floorY + ROBOT_UP.y * 1.15;
  return { anchor, center, floorY };
}

function transformVisualPoint(point, transform) {
  const rotation = rotationRowsToMatrix4(normalizeRotationRows(transform.rotationMatrix));
  return point.clone().multiplyScalar(transform.scale).applyMatrix4(rotation).add(transform.translation);
}

function transformVisualDirection(direction, transform) {
  const rotation = rotationRowsToMatrix4(normalizeRotationRows(transform.rotationMatrix));
  return direction.clone().applyMatrix4(rotation).normalize();
}

function recordedCameraPreset(presetId) {
  const preset = manifest?.cameraPresets?.[presetId];
  if (!preset || preset.coordinateFrame !== "visual_native" || !baseVisualTransform) return null;
  if (!Array.isArray(preset.eye) || !Array.isArray(preset.forward)) return null;
  const transform = applyScenePresentationToVisualTransform(baseVisualTransform);
  const nativeEye = vectorFromArray(preset.eye);
  const nativeForward = vectorFromArray(preset.forward);
  const nativeImageUp = Array.isArray(preset.up)
    ? vectorFromArray(preset.up).multiplyScalar(-1)
    : new THREE.Vector3(0, 1, 0);
  if (nativeForward.lengthSq() < 1e-8) return null;
  nativeForward.normalize();
  const targetDistance = clampFinite(preset.targetDistance, 0.5, 40, 6);
  const nativeTarget = nativeEye.clone().addScaledVector(nativeForward, targetDistance);
  return {
    eye: transformVisualPoint(nativeEye, transform),
    target: transformVisualPoint(nativeTarget, transform),
    up: transformVisualDirection(nativeImageUp, transform),
    fovDeg: clampFinite(preset.verticalFovDeg, 32, 95, DEFAULT_CAMERA_FOV_DEG),
    source: preset.id || preset.sourceFrame || "recorded-camera",
  };
}

function cameraPresetVectors(preset, bounds) {
  const size = boxSize(bounds);
  const roomMax = Math.max(size.x, size.y, size.z, 1);
  let presetId = CAMERA_PRESETS.includes(preset) ? preset : "reference";
  const recorded = recordedCameraPreset(presetId);
  if (recorded) return { ...recorded, size, roomMax, presetId };
  if (presetId === "doorway") presetId = "back";
  const { anchor, center, floorY } = cameraAnchorFromScene(bounds, { preferRobot: presetId === "robot" });
  const indoorDistance = Math.max(2.6, Math.min(7.2, Math.max(size.x, size.z) * 0.18));
  const lowHeight = Math.max(1.15, Math.min(2.1, size.y * 0.14));
  const lookAhead = Math.max(1.2, Math.min(3.8, Math.max(size.x, size.z) * 0.055));
  const upSign = Math.sign(ROBOT_UP.y) || 1;
  let eye;
  let target = anchor.clone().addScaledVector(ROBOT_UP, 0.42);

  if (presetId === "front") {
    eye = anchor.clone().add(new THREE.Vector3(0, upSign * lowHeight, -indoorDistance));
    target.add(new THREE.Vector3(0, upSign * 0.05, lookAhead));
  } else if (presetId === "back") {
    eye = anchor.clone().add(new THREE.Vector3(0, upSign * lowHeight, indoorDistance));
    target.add(new THREE.Vector3(0, upSign * 0.05, -lookAhead));
  } else if (presetId === "left") {
    eye = anchor.clone().add(new THREE.Vector3(-indoorDistance, upSign * lowHeight, 0));
    target.add(new THREE.Vector3(lookAhead, upSign * 0.05, 0));
  } else if (presetId === "right") {
    eye = anchor.clone().add(new THREE.Vector3(indoorDistance, upSign * lowHeight, 0));
    target.add(new THREE.Vector3(-lookAhead, upSign * 0.05, 0));
  } else if (presetId === "top") {
    const topDistance = Math.max(6.5, Math.min(18, roomMax * 0.46));
    target = new THREE.Vector3(center.x, floorY + upSign * 0.45, center.z);
    eye = target.clone().addScaledVector(ROBOT_UP, topDistance).add(new THREE.Vector3(0.02, 0, 0.02));
  } else {
    const yaw = robotGroup ? robotGroup.rotation.y : 0;
    const back = new THREE.Vector3(Math.sin(yaw + Math.PI), 0, Math.cos(yaw + Math.PI));
    const side = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw)).multiplyScalar(0.65);
    target = anchor.clone().addScaledVector(ROBOT_UP, 0.52);
    eye = target.clone()
      .addScaledVector(back, Math.max(2.4, Math.min(4.2, indoorDistance * 0.74)))
      .add(side)
      .addScaledVector(ROBOT_UP, Math.max(1.05, lowHeight * 0.75));
  }

  const minEyeY = floorY + upSign * 0.35;
  if ((eye.y - minEyeY) * upSign < 0) eye.y = minEyeY;
  return {
    eye,
    target,
    size,
    roomMax,
    presetId,
    fovDeg: DEFAULT_CAMERA_FOV_DEG,
    source: `procedural-${presetId}`,
    up: ROBOT_UP.clone(),
  };
}

function setCameraOrbitView(preset = "reference", { announce = false, immediate = false } = {}) {
  const bounds = transformedVisualBounds || colliderBounds;
  if (!bounds || bounds.isEmpty()) return;
  const { eye, target, roomMax, presetId, fovDeg, source, up } = cameraPresetVectors(preset, bounds);
  state.robotFollowCamera = false;
  state.cameraMode = "orbit";
  state.cameraPreset = presetId;
  state.cameraPresetSource = source;
  keyState.clear();
  controls.minDistance = 0.08;
  controls.maxDistance = Math.max(16, Math.min(48, roomMax * 1.35));
  camera.near = 0.02;
  camera.far = Math.max(220, roomMax * 12);
  camera.fov = fovDeg;
  camera.up.copy(up || ROBOT_UP);
  camera.updateProjectionMatrix();
  const overviewId = manifest?.initialState?.cameraFocusObjectId;
  const overview = overviewId ? interactiveObjects.get(String(overviewId)) : null;
  if (overview) {
    focusCameraOnInteractiveComponent(overview, {
      immediate,
      referenceEye: eye,
      referenceDirection: eye.clone().sub(target),
      overview: true,
      preserveVertical: presetId === "top",
      preservePresetDirection: true,
    });
    state.cameraPresetSource = `${source}+overview:${overview.definition.id}`;
  } else {
    state.cameraOverviewFocusObjectId = null;
    setSmoothCameraPose(eye, target, { immediate });
  }
  setLayerVisibility();
  if (announce) showToast(`Camera preset: ${CAMERA_PRESET_LABELS[presetId]}.`);
}

function frameCameraToInterior({ announce = false } = {}) {
  const nextPreset = announce ? nextValue(CAMERA_PRESETS, state.cameraPreset) : state.cameraPreset;
  setCameraOrbitView(nextPreset || "reference", { announce });
}

function updateAdaptiveQuality(fps) {
  if (!state.adaptivePixelRatio || state.qualityMode !== "balanced") return;
  const current = renderer.getPixelRatio();
  const ceiling = qualityPixelRatioLimit();
  if (fps < 24 && current > 0.78) {
    applyRendererPixelRatio(Math.max(0.8, current - 0.12), { forceResize: true });
    return;
  }
  if (fps > 48 && current < ceiling - 0.06) {
    applyRendererPixelRatio(Math.min(ceiling, current + 0.06), { forceResize: true });
  }
}

function markHit(point, normal) {
  markerLayer.clear();
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.22, 20, 20),
    new THREE.MeshStandardMaterial({
      color: 0xff806d,
      emissive: 0x6b1109,
      emissiveIntensity: 0.8,
    })
  );
  marker.position.copy(point);
  markerLayer.add(marker);
  const arrow = new THREE.ArrowHelper(normal, point, 1.4, 0xff806d, 0.36, 0.18);
  markerLayer.add(arrow);
}

function describeHit(hit, normal) {
  const faceIndex = Number.isFinite(hit.faceIndex) ? hit.faceIndex : -1;
  const semantic = colliderFaceSemantics[faceIndex] || {};
  return {
    label: hit.object.userData.colliderLabel || hit.object.name || "collider",
    colliderKind: hit.object.userData.colliderKind || "scene",
    colliderId: hit.object.userData.colliderId || "scene",
    surfaceType: hit.object.userData.surfaceType || "mesh-collider",
    faceIndex,
    objectId: hit.object.userData.interactiveObjectId
      || (Number.isFinite(semantic.objectId) ? semantic.objectId : null),
    objectProbability: Number.isFinite(semantic.probability)
      ? Number(semantic.probability.toFixed(4))
      : null,
    sourceFace: Number.isFinite(semantic.sourceFace) ? semantic.sourceFace : null,
    point: {
      x: Number(hit.point.x.toFixed(4)),
      y: Number(hit.point.y.toFixed(4)),
      z: Number(hit.point.z.toFixed(4)),
    },
    normal: {
      x: Number(normal.x.toFixed(4)),
      y: Number(normal.y.toFixed(4)),
      z: Number(normal.z.toFixed(4)),
    },
    distance: Number(hit.distance.toFixed(4)),
  };
}

function setRaycasterFromCanvasPoint(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
}

function disposeSceneEntityOutline() {
  if (!sceneEntitySelectionOutline) return;
  sceneEntityLayer.remove(sceneEntitySelectionOutline);
  sceneEntitySelectionOutline.geometry?.dispose?.();
  sceneEntitySelectionOutline.material?.dispose?.();
  sceneEntitySelectionOutline = null;
}

function showSceneEntityBounds(entity) {
  disposeSceneEntityOutline();
  if (!entity?.bbox) return null;
  const size = vectorFromArray(entity.bbox.extent, [0.1, 0.1, 0.1]);
  const center = vectorFromArray(entity.bbox.center);
  const boxGeometry = new THREE.BoxGeometry(
    Math.max(size.x, 0.04),
    Math.max(size.y, 0.04),
    Math.max(size.z, 0.04)
  );
  sceneEntitySelectionOutline = new THREE.LineSegments(
    new THREE.EdgesGeometry(boxGeometry),
    new THREE.LineBasicMaterial({ color: 0xefb35f, transparent: true, opacity: 0.96 })
  );
  boxGeometry.dispose();
  sceneEntitySelectionOutline.name = `${entity.id} scene-query bounds`;
  sceneEntitySelectionOutline.position.copy(center);
  sceneEntitySelectionOutline.raycast = () => {};
  sceneEntityLayer.add(sceneEntitySelectionOutline);
  sceneEntityLayer.updateMatrixWorld(true);
  return sceneEntitySelectionOutline;
}

function focusCameraOnBox(worldBox) {
  if (!worldBox || worldBox.isEmpty() || state.cameraMode !== "orbit") return;
  const worldCenter = worldBox.getCenter(new THREE.Vector3());
  const dimensions = worldBox.getSize(new THREE.Vector3());
  const roomCenter = colliderBounds ? boxCenter(colliderBounds) : controls.target.clone();
  const inward = roomCenter.sub(worldCenter);
  inward.addScaledVector(ROBOT_UP, -inward.dot(ROBOT_UP));
  if (inward.lengthSq() < 1e-5) {
    inward.copy(camera.position).sub(worldCenter);
    inward.addScaledVector(ROBOT_UP, -inward.dot(ROBOT_UP));
  }
  inward.normalize();
  const distance = Math.max(5.5, Math.max(dimensions.x, dimensions.y, dimensions.z) * 1.85);
  const eye = worldCenter.clone()
    .addScaledVector(inward, distance)
    .addScaledVector(ROBOT_UP, Math.max(0.9, dimensions.y * 0.28));
  setSmoothCameraPose(eye, worldCenter, { immediate: false });
}

function focusSceneEntity(entityOrId, { focus = true } = {}) {
  const entity = typeof entityOrId === "string"
    ? sceneKnowledgeIndex?.entities.get(entityOrId)
    : entityOrId;
  if (!entity) return false;
  state.selectedSceneEntity = entity.id;
  const componentId = entity.interactiveObjectId || entity.id;
  const component = interactiveObjects.get(componentId);
  if (component) {
    disposeSceneEntityOutline();
    selectInteractiveObject(component, { focus });
  } else if (entity.bbox) {
    showSceneEntityBounds(entity);
    const localBox = new THREE.Box3(
      vectorFromArray(entity.bbox.min),
      vectorFromArray(entity.bbox.max)
    );
    sceneEntityLayer.updateMatrixWorld(true);
    if (focus) focusCameraOnBox(localBox.applyMatrix4(sceneEntityLayer.matrixWorld));
  }
  updateHud();
  return true;
}

function renderSceneQaResult(result) {
  if (!sceneQaAnswer || !sceneQaCandidates) return;
  sceneQaAnswer.textContent = result.answer || "";
  sceneQaAnswer.dataset.status = result.status || "idle";
  sceneQaCandidates.replaceChildren();
  for (const entity of result.candidates || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = entity.name || entity.label || entity.id;
    button.dataset.sceneEntityId = entity.id;
    button.addEventListener("click", () => {
      const suffix = state.sceneQaIntent === "appearance" ? " appearance" : " where";
      const question = `${entity.id}${suffix}`;
      if (sceneQaInput) sceneQaInput.value = question;
      askScene(question);
    });
    sceneQaCandidates.append(button);
  }
}

function clearSceneCommandPreview({ resetState = true } = {}) {
  currentSceneCommandPreview = null;
  currentSceneCommandSubmission = null;
  if (sceneCommandPreviewElement) sceneCommandPreviewElement.hidden = true;
  if (sceneCommandTitle) sceneCommandTitle.textContent = "";
  if (sceneCommandRisk) sceneCommandRisk.textContent = "";
  if (sceneCommandSummary) sceneCommandSummary.textContent = "";
  if (sceneCommandStages) sceneCommandStages.replaceChildren();
  if (sceneCommandStatus) {
    sceneCommandStatus.textContent = "";
    sceneCommandStatus.dataset.status = "idle";
  }
  if (sceneCommandConfirm) {
    sceneCommandConfirm.disabled = true;
    sceneCommandConfirm.textContent = "确认提交";
  }
  if (!resetState) return;
  state.sceneCommandReady = false;
  state.sceneCommandKind = null;
  state.sceneCommandStatus = "idle";
  state.sceneCommandAffectedStages = [];
  state.sceneCommandTargetIds = [];
  state.sceneCommandRequestId = null;
  state.sceneCommandPlanStatus = "idle";
  state.sceneCommandConfirmationRequired = false;
  state.sceneCommandResponse = null;
  state.sceneCommandError = "";
}

function setSceneCommandStatus(message, status) {
  if (sceneCommandStatus) {
    sceneCommandStatus.textContent = message;
    sceneCommandStatus.dataset.status = status;
  }
  state.sceneCommandStatus = status;
}

function renderSceneCommandPreview(preview) {
  currentSceneCommandPreview = preview;
  if (sceneQaAnswer) {
    sceneQaAnswer.textContent = "";
    sceneQaAnswer.dataset.status = "idle";
  }
  if (sceneQaCandidates) sceneQaCandidates.replaceChildren();
  if (sceneCommandPreviewElement) sceneCommandPreviewElement.hidden = false;
  if (sceneCommandTitle) sceneCommandTitle.textContent = `${preview.kindLabel} · 只读预览`;
  if (sceneCommandRisk) sceneCommandRisk.textContent = preview.riskLevel;
  if (sceneCommandSummary) sceneCommandSummary.textContent = preview.summary || "";
  if (sceneCommandStages) {
    sceneCommandStages.replaceChildren();
    for (const affected of preview.affectedStages || []) {
      const chip = document.createElement("span");
      chip.textContent = affected.stage;
      sceneCommandStages.append(chip);
    }
  }

  state.sceneCommandReady = preview.status === "ready";
  state.sceneCommandKind = preview.kind;
  state.sceneCommandPreviewOnly = true;
  state.sceneCommandAffectedStages = (preview.affectedStages || []).map((item) => item.stage);
  state.sceneCommandTargetIds = [...(preview.targetIds || [])];
  state.sceneCommandRequestId = null;
  state.sceneCommandResponse = null;
  state.sceneCommandError = "";
  if (preview.status !== "ready") {
    setSceneCommandStatus("目标未通过确定性解析，操作已阻止。", "blocked");
    if (sceneCommandConfirm) sceneCommandConfirm.disabled = true;
  } else if (!sceneCommandEndpoint) {
    setSceneCommandStatus("当前 manifest 未配置执行后端；这里只显示预览，场景不会被修改。", "blocked");
    if (sceneCommandConfirm) sceneCommandConfirm.disabled = true;
  } else {
    setSceneCommandStatus("正在请求后端生成权威执行计划…", "submitting");
    if (sceneCommandConfirm) sceneCommandConfirm.disabled = true;
  }
  updateHud();
  return preview;
}

function previewSceneInput(question) {
  if (!sceneKnowledgeIndex) {
    return renderSceneCommandPreview({
      kind: "query_description",
      kindLabel: "场景指令",
      mutating: true,
      status: "blocked_invalid",
      previewOnly: true,
      affectedStages: [],
      targetIds: [],
      riskLevel: "none",
      summary: "场景知识尚未就绪。",
    });
  }
  const preview = parseSceneCommand(question, sceneKnowledgeIndex, {
    selectedEntityId: state.selectedSceneEntity,
  });
  if (!preview.mutating) {
    clearSceneCommandPreview();
    return askScene(question);
  }
  const rendered = renderSceneCommandPreview(preview);
  if (preview.status === "ready" && sceneCommandEndpoint) {
    prepareSceneCommandSubmission(preview);
  }
  return rendered;
}

function sceneCommandRequestId() {
  const random = globalThis.crypto?.randomUUID?.().replaceAll("-", "")
    || Math.random().toString(36).slice(2, 14);
  return `web-${Date.now()}-${random}`;
}

function expectedManifestSha256() {
  const candidates = [
    manifest?.expectedManifestSha256,
    manifest?.worldManifestSha256,
    manifest?.sourceWorld?.manifestSha256,
    manifest?.sourceWorld?.sha256,
  ];
  return candidates.find((value) => typeof value === "string" && /^[a-f0-9]{64}$/u.test(value)) || null;
}

function sceneCommandQueuePayload(command, preview) {
  return {
    command,
    structuredIntent: structuredClone(command.intent),
    rawPrompt: command.provenance.raw_prompt,
    expectedManifestSha256: expectedManifestSha256(),
    clientPreview: {
      status: preview.status,
      previewOnly: true,
      manifestVersion: state.manifestVersion || null,
      riskLevel: preview.riskLevel,
      summary: preview.summary,
      targetIds: [...preview.targetIds],
      affectedStages: preview.affectedStages.map((item) => item.stage),
    },
  };
}

function sceneCommandPlanEndpoint(submitEndpoint) {
  const endpoint = new URL(submitEndpoint);
  if (!endpoint.pathname.endsWith("/v1/scene-commands/submit")) {
    throw new Error("Scene command endpoint does not expose the v1 plan route");
  }
  endpoint.pathname = endpoint.pathname.replace(/\/submit$/u, "/plan");
  return endpoint.href;
}

async function prepareSceneCommandSubmission(preview) {
  const requestId = sceneCommandRequestId();
  const command = buildSceneCommandEnvelope(preview, {
    requestId,
    contextReferences: [
      `manifest:${state.manifestVersion || "unknown"}`,
      ...(expectedManifestSha256() ? [`manifest-sha256:${expectedManifestSha256()}`] : []),
    ],
  });
  const payload = sceneCommandQueuePayload(command, preview);
  currentSceneCommandSubmission = {
    requestId,
    command,
    payload,
    confirmationPhrase: null,
    planResult: null,
  };
  state.sceneCommandRequestId = requestId;
  state.sceneCommandPlanStatus = "planning";
  state.sceneCommandConfirmationRequired = false;
  try {
    const result = await planSceneCommandEnvelope(
      sceneCommandPlanEndpoint(sceneCommandEndpoint),
      payload,
    );
    if (currentSceneCommandSubmission?.requestId !== requestId) return result;
    currentSceneCommandSubmission.planResult = structuredClone(result);
    currentSceneCommandSubmission.confirmationPhrase = result.confirmationPhrase || null;
    state.sceneCommandPlanStatus = result.status;
    state.sceneCommandConfirmationRequired = result.confirmationRequired;
    state.sceneCommandResponse = structuredClone(result);
    const planReady = result.status === "ready" && result.plan?.status === "ready";
    if (!planReady) {
      const reason = result.plan?.warnings?.[0] || result.reason || "服务端计划未通过。";
      setSceneCommandStatus(`权威计划阻止了该操作：${reason}`, "blocked");
      if (sceneCommandConfirm) sceneCommandConfirm.disabled = true;
      return result;
    }
    const planStages = Array.isArray(result.plan?.affected_stages)
      ? result.plan.affected_stages.map((item) => item.stage).filter(Boolean)
      : [];
    if (planStages.length) {
      state.sceneCommandAffectedStages = planStages;
      if (sceneCommandStages) {
        sceneCommandStages.replaceChildren();
        for (const stage of planStages) {
          const chip = document.createElement("span");
          chip.textContent = stage;
          sceneCommandStages.append(chip);
        }
      }
    }
    setSceneCommandStatus(
      result.confirmationRequired
        ? "后端计划已验证；确认后将携带服务端签发的确认短语入队。"
        : "后端计划已验证；确认后入队。",
      "ready",
    );
    if (sceneCommandConfirm) sceneCommandConfirm.disabled = false;
    return result;
  } catch (error) {
    if (currentSceneCommandSubmission?.requestId !== requestId) return null;
    state.sceneCommandPlanStatus = "failed";
    state.sceneCommandError = error?.message || String(error);
    setSceneCommandStatus(`计划校验失败：${state.sceneCommandError}`, "failed");
    if (sceneCommandConfirm) sceneCommandConfirm.disabled = true;
    return null;
  } finally {
    updateHud();
  }
}

async function submitCurrentSceneCommand() {
  const preview = currentSceneCommandPreview;
  const submission = currentSceneCommandSubmission;
  if (
    !preview
    || preview.status !== "ready"
    || !sceneCommandEndpoint
    || !submission?.planResult
    || submission.planResult.status !== "ready"
  ) {
    return { status: "blocked", reason: "no_ready_preview_or_execution_backend" };
  }
  const payload = {
    ...submission.payload,
    ...(submission.confirmationPhrase
      ? { confirmationPhrase: submission.confirmationPhrase }
      : {}),
  };
  state.sceneCommandRequestId = submission.requestId;
  state.sceneCommandError = "";
  if (sceneCommandConfirm) {
    sceneCommandConfirm.disabled = true;
    sceneCommandConfirm.textContent = "提交中";
  }
  setSceneCommandStatus("正在提交结构化指令…", "submitting");
  updateHud();
  try {
    const result = await submitSceneCommandEnvelope(sceneCommandEndpoint, payload);
    state.sceneCommandResponse = structuredClone(result);
    if (result.status === "queued") {
      setSceneCommandStatus("已进入 pipeline 队列；当前页面中的场景资产尚未改变。", "accepted");
    } else if (result.status === "completed_read") {
      setSceneCommandStatus("后端已完成只读请求；当前页面中的场景资产未改变。", "accepted");
    } else if (result.status === "duplicate") {
      setSceneCommandStatus("同一请求已经存在，未重复入队。", "accepted");
    } else {
      const reason = result.reason || result.message || "后端阻止了该操作。";
      setSceneCommandStatus(`未入队：${reason}`, "blocked");
    }
    return result;
  } catch (error) {
    state.sceneCommandError = error?.message || String(error);
    state.sceneCommandResponse = null;
    setSceneCommandStatus(`提交失败：${state.sceneCommandError}`, "failed");
    return { status: "blocked", reason: state.sceneCommandError };
  } finally {
    if (sceneCommandConfirm) {
      sceneCommandConfirm.disabled = true;
      sceneCommandConfirm.textContent = "已提交";
    }
    updateHud();
  }
}

function askScene(question) {
  clearSceneCommandPreview();
  if (!sceneKnowledgeIndex) {
    const unavailable = {
      status: "unavailable",
      answer: "Scene knowledge is not ready.",
      candidates: [],
      focusEntityId: null,
    };
    renderSceneQaResult(unavailable);
    return unavailable;
  }
  const result = answerSceneQuestion(question, sceneKnowledgeIndex, {
    selectedEntityId: state.selectedSceneEntity,
  });
  state.sceneQaQuery = String(question || "");
  state.sceneQaIntent = result.intent || null;
  state.sceneQaStatus = result.status;
  state.sceneQaAnswer = result.answer || "";
  state.sceneQaResolvedObjectId = result.entity?.id || null;
  state.sceneQaCandidates = (result.candidates || []).map((entity) => entity.id);
  state.sceneQaError = "";
  renderSceneQaResult(result);
  if (result.focusEntityId) focusSceneEntity(result.focusEntityId, { focus: true });
  else updateHud();
  return {
    status: result.status,
    intent: result.intent,
    answer: result.answer,
    entityId: result.entity?.id || null,
    focusEntityId: result.focusEntityId,
    candidates: (result.candidates || []).map((entity) => ({ id: entity.id, name: entity.name })),
  };
}

function interactiveObjectHitAt(clientX, clientY) {
  if (!interactiveObjectColliders.length) return null;
  setRaycasterFromCanvasPoint(clientX, clientY);
  const selectHit = (hits) => {
    if (!state.interactiveObjectSolo) return hits[0] || null;
    return hits.find((hit) => (
      hit.object.userData.interactiveObjectId === state.selectedInteractiveObject
    )) || null;
  };
  const frontSideHit = selectHit(raycaster.intersectObjects(interactiveObjectColliders, false));
  if (frontSideHit) return frontSideHit;

  // Surface-BVH assets can be valid open surfaces. Make picking two-sided without
  // changing their retained PBR render materials or using a proxy geometry.
  const unifiedMeshes = interactiveObjectColliders.filter(
    (mesh) => mesh.userData.colliderMode === "unified-glb",
  );
  const originalSides = [];
  for (const mesh of unifiedMeshes) {
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const material of materials) {
      if (!material) continue;
      originalSides.push([material, material.side]);
      material.side = THREE.DoubleSide;
    }
  }
  const doubleSideHit = selectHit(raycaster.intersectObjects(unifiedMeshes, false));
  for (const [material, side] of originalSides) material.side = side;
  return doubleSideHit;
}

function setComponentVisualVisibility(component, visible) {
  if (!component?.splat) return;
  component.splat.visible = Boolean(visible);
  if (component.visualKind === "gaussian-splat") {
    component.splat.opacity = visible ? 1 : 0;
  } else if (component.splat.material) {
    component.splat.material.opacity = visible ? 1 : 0;
    component.splat.material.needsUpdate = true;
  }
}

function interactiveObjectVisibility(component) {
  if (!state.interactiveObjectSolo) return { groupVisible: true, objectVisible: true };
  const selected = interactiveObjects.get(state.selectedInteractiveObject);
  if (!selected) return { groupVisible: false, objectVisible: false };
  if (component === selected) return { groupVisible: true, objectVisible: true };
  let ancestor = selected.parentComponent;
  while (ancestor) {
    if (ancestor === component) return { groupVisible: true, objectVisible: false };
    ancestor = ancestor.parentComponent;
  }
  return { groupVisible: false, objectVisible: false };
}

function applyInteractiveObjectVisibility() {
  for (const component of interactiveObjects.values()) {
    const { groupVisible, objectVisible } = interactiveObjectVisibility(component);
    const selected = state.selectedInteractiveObject === component.definition.id;
    component.group.visible = groupVisible;
    setComponentVisualVisibility(component, state.showVisual && objectVisible);
    if (component.proxy?.material) {
      component.proxy.material.opacity = state.interactiveObjectProxiesVisible && objectVisible ? 0.07 : 0;
    }
    component.outline.visible = objectVisible && (
      state.interactiveObjectProxiesVisible || selected || Boolean(component.spin)
    );
  }
}

function focusCameraOnInteractiveComponent(component, {
  immediate = false,
  overview = false,
  referenceEye = null,
  referenceDirection = null,
  preserveVertical = null,
  preservePresetDirection = false,
} = {}) {
  if (!component || state.cameraMode !== "orbit") return false;
  const worldBounds = interactiveObjectCollisionBounds(component);
  const worldPosition = worldBounds
    ? worldBounds.getCenter(new THREE.Vector3())
    : component.group.getWorldPosition(new THREE.Vector3());
  const dimensions = worldBounds
    ? worldBounds.getSize(new THREE.Vector3()).toArray()
    : vectorFromArray(component.definition.colliderProxy?.dimensions, [0.5, 0.5, 0.5]).toArray();
  const boundingRadius = Math.hypot(...dimensions) * 0.5;
  const verticalHalfFov = THREE.MathUtils.degToRad(camera.fov) * 0.5;
  const horizontalHalfFov = Math.atan(Math.tan(verticalHalfFov) * Math.max(camera.aspect, 0.1));
  const framingHalfFov = Math.max(THREE.MathUtils.degToRad(8), Math.min(verticalHalfFov, horizontalHalfFov));
  const framingDistance = boundingRadius / Math.max(Math.sin(framingHalfFov), 1e-3) * 1.08;
  const distance = Math.max(5.5, Math.max(...dimensions) * 1.85, framingDistance);
  controls.maxDistance = Math.max(controls.maxDistance, distance * 1.05);
  const roomCenter = colliderBounds ? boxCenter(colliderBounds) : controls.target.clone();
  const currentDirection = referenceDirection?.clone()
    || (referenceEye || camera.position).clone().sub(worldPosition);
  const inward = roomCenter.sub(worldPosition);
  const normalizedUp = ROBOT_UP.clone().normalize();
  const currentUnit = currentDirection.lengthSq() > 1e-8
    ? currentDirection.clone().normalize()
    : normalizedUp.clone();
  const verticalView = preserveVertical == null
    ? Math.abs(currentUnit.dot(normalizedUp)) >= 0.82
    : preserveVertical;
  const flatten = (direction) => direction.addScaledVector(ROBOT_UP, -direction.dot(ROBOT_UP));
  if (!verticalView) flatten(currentDirection);
  flatten(inward);
  if (currentDirection.lengthSq() < 1e-5) currentDirection.copy(inward);
  if (inward.lengthSq() < 1e-5) {
    inward.copy(new THREE.Vector3(0, 0, 1)).applyAxisAngle(ROBOT_UP, Math.PI / 4);
  }
  currentDirection.normalize();
  inward.normalize();
  const candidateDirections = [];
  const appendOrbitDirections = (seed, angles = [0, Math.PI / 4, -Math.PI / 4, Math.PI / 2, -Math.PI / 2, Math.PI]) => {
    for (const angle of angles) {
      candidateDirections.push(seed.clone().applyAxisAngle(ROBOT_UP, angle).normalize());
    }
  };
  if (verticalView) {
    candidateDirections.push(currentUnit);
  } else if (preservePresetDirection) {
    appendOrbitDirections(currentDirection, [0, Math.PI / 8, -Math.PI / 8, Math.PI / 4, -Math.PI / 4]);
  } else {
    appendOrbitDirections(currentDirection);
  }
  if (!preservePresetDirection) appendOrbitDirections(inward);
  const height = verticalView ? 0 : Math.max(0.9, dimensions[1] * 0.28);
  const previousFar = raycaster.far;
  interactiveObjectLayer.updateMatrixWorld(true);
  const expandedCollisionBounds = Array.from(interactiveObjects.values(), (candidate) =>
    interactiveObjectCollisionBounds(candidate)?.clone().expandByScalar(0.05)
  ).filter(Boolean);
  const hitsFocusedAssembly = (hit) => {
    let hitComponent = interactiveObjects.get(hit?.object?.userData?.interactiveObjectId);
    while (hitComponent) {
      if (hitComponent === component) return true;
      hitComponent = hitComponent.parentComponent;
    }
    return false;
  };
  const eyeCandidates = candidateDirections.map((direction) => worldPosition.clone()
    .addScaledVector(direction, distance)
    .addScaledVector(ROBOT_UP, height));
  const eye = eyeCandidates.find((candidateEye) => {
    if (expandedCollisionBounds.some((bounds) => bounds.containsPoint(candidateEye))) return false;
    const line = worldPosition.clone().sub(candidateEye);
    const lineDistance = line.length();
    raycaster.set(candidateEye, line.normalize());
    raycaster.far = lineDistance * 1.05;
    const firstHit = raycaster.intersectObjects(interactiveObjectColliders, false)[0];
    return hitsFocusedAssembly(firstHit);
  }) || eyeCandidates.find((candidateEye) =>
    !expandedCollisionBounds.some((bounds) => bounds.containsPoint(candidateEye))
  ) || eyeCandidates[0];
  raycaster.far = previousFar;
  state.cameraOverviewFocusObjectId = overview ? component.definition.id : null;
  setSmoothCameraPose(eye, worldPosition, { immediate });
  return true;
}

function selectInteractiveObject(component, { focus = false } = {}) {
  if (!component) return;
  state.selectedInteractiveObject = component.definition.id;
  state.selectedSceneEntity = component.definition.id;
  for (const candidate of interactiveObjects.values()) {
    const selected = candidate === component;
    candidate.outline.material.color.setHex(selected ? 0xefb35f : 0x58d7c9);
  }
  applyInteractiveObjectVisibility();
  if (focus) focusCameraOnInteractiveComponent(component);
  updateHud();
}

function startInteractiveObjectSpin(component) {
  if (
    !component
    || component.definition.independentlyMovable !== true
    || component.definition.interaction?.kind !== "spin"
    || component.spin
    || objectDrag?.component === component
  ) return false;
  const duration = Math.max(0.35, Number(component.definition.interaction?.durationMs || 1250) / 1000);
  component.spin = {
    elapsed: 0,
    duration,
    startQuaternion: component.group.quaternion.clone(),
  };
  state.interactiveObjectRotationActive = true;
  selectInteractiveObject(component, { focus: true });
  const visualLabel = component.visualKind === "rgb-points"
    ? "RGB point visual"
    : component.visualKind === "mesh" ? "mesh visual" : "Gaussian visual";
  const proxyLabel = component.collisionMode === "unified-glb"
    ? "shared PBR mesh collider"
    : component.collisionMode === "none" ? "selection bounds" : "mesh collider";
  showToast(`${component.definition.label}: ${visualLabel} and ${proxyLabel} rotating together.`);
  return true;
}

function setInteractiveObjectYaw(component, degrees) {
  if (!component || component.definition.independentlyMovable !== true) return false;
  const radians = THREE.MathUtils.degToRad(Number(degrees) || 0);
  const localUp = new THREE.Vector3(0, Math.sign(ROBOT_UP.y) || 1, 0);
  const rotation = new THREE.Quaternion().setFromAxisAngle(localUp, radians);
  component.group.quaternion.copy(component.restQuaternion).multiply(rotation);
  component.group.updateMatrixWorld(true);
  component.dragYawRadians = radians;
  state.objectDragYawDeg = Number(THREE.MathUtils.radToDeg(radians).toFixed(3));
  return true;
}

function beginInteractiveObjectDrag(event, hit) {
  const objectId = hit?.object?.userData?.interactiveObjectId;
  const component = interactiveObjects.get(objectId);
  if (
    !component
    || component.definition.independentlyMovable !== true
    || component.definition.interaction?.drag !== "horizontal_yaw"
    || component.spin
  ) return false;
  event.preventDefault();
  state.robotFollowCamera = false;
  selectInteractiveObject(component, { focus: false });
  objectDrag = {
    pointerId: event.pointerId,
    component,
    startX: event.clientX,
    startY: event.clientY,
    lastX: event.clientX,
    lastY: event.clientY,
    moved: false,
  };
  state.activeObjectDrag = component.definition.id;
  canvas.style.cursor = "grabbing";
  canvas.setPointerCapture?.(event.pointerId);
  return true;
}

function updateInteractiveObjectDrag(event) {
  if (!objectDrag || event.pointerId !== objectDrag.pointerId) return false;
  event.preventDefault();
  const dx = event.clientX - objectDrag.lastX;
  objectDrag.lastX = event.clientX;
  objectDrag.lastY = event.clientY;
  if (Math.hypot(event.clientX - objectDrag.startX, event.clientY - objectDrag.startY) > 4) {
    objectDrag.moved = true;
  }
  if (dx !== 0) {
    const localUp = new THREE.Vector3(0, Math.sign(ROBOT_UP.y) || 1, 0);
    const delta = new THREE.Quaternion().setFromAxisAngle(localUp, -dx * OBJECT_DRAG_YAW_PER_PIXEL);
    objectDrag.component.group.quaternion.multiply(delta);
    objectDrag.component.group.updateMatrixWorld(true);
    objectDrag.component.dragYawRadians -= dx * OBJECT_DRAG_YAW_PER_PIXEL;
    state.objectDragYawDeg = Number(THREE.MathUtils.radToDeg(objectDrag.component.dragYawRadians).toFixed(3));
  }
  return true;
}

function endInteractiveObjectDrag(event, { cancelled = false } = {}) {
  if (!objectDrag || (event?.pointerId != null && event.pointerId !== objectDrag.pointerId)) return false;
  const completed = objectDrag;
  try {
    canvas.releasePointerCapture?.(completed.pointerId);
  } catch {
    // Pointer capture may already be released by the browser.
  }
  objectDrag = null;
  state.activeObjectDrag = null;
  canvas.style.cursor = "";
  completed.component.group.updateMatrixWorld(true);
  if (!cancelled && !completed.moved) selectInteractiveObject(completed.component, { focus: false });
  updateHud();
  return true;
}

function updateInteractiveObjectAnimations(dt) {
  let active = false;
  const localUp = new THREE.Vector3(0, Math.sign(ROBOT_UP.y) || 1, 0);
  for (const component of interactiveObjects.values()) {
    if (!component.spin) continue;
    active = true;
    component.spin.elapsed = Math.min(component.spin.duration, component.spin.elapsed + dt);
    const progress = component.spin.elapsed / component.spin.duration;
    const eased = progress < 0.5
      ? 4 * progress * progress * progress
      : 1 - Math.pow(-2 * progress + 2, 3) / 2;
    const spinQuaternion = new THREE.Quaternion().setFromAxisAngle(localUp, eased * Math.PI * 2);
    component.group.quaternion.copy(component.spin.startQuaternion).multiply(spinQuaternion);
    component.group.updateMatrixWorld(true);
    if (progress >= 1) {
      component.group.quaternion.copy(component.spin.startQuaternion);
      component.group.updateMatrixWorld(true);
      component.spin = null;
      component.turns += 1;
      state.interactiveObjectTurns += 1;
      component.outline.visible = state.interactiveObjectProxiesVisible || state.selectedInteractiveObject === component.definition.id;
      showToast(`${component.definition.label} completed one 360° turn.`);
    }
  }
  state.interactiveObjectRotationActive = active || Array.from(interactiveObjects.values()).some((component) => component.spin);
}

function focusNextInteractiveObject() {
  const ready = Array.from(interactiveObjects.values()).filter((component) => component.ready);
  if (!ready.length) {
    showToast("Interactive objects are still loading.");
    return;
  }
  const currentIndex = ready.findIndex((component) => component.definition.id === state.selectedInteractiveObject);
  const component = ready[(currentIndex + 1 + ready.length) % ready.length];
  selectInteractiveObject(component, { focus: true });
  showToast(`Selected ${component.definition.label}. Double-click its proxy to spin.`);
}

function onCanvasDoubleClick(event) {
  if (event.button !== 0) return;
  event.preventDefault();
  const hit = interactiveObjectHitAt(event.clientX, event.clientY);
  if (!hit) {
    showToast("No interactive object proxy under the pointer.");
    return;
  }
  const objectId = hit.object.userData.interactiveObjectId;
  startInteractiveObjectSpin(interactiveObjects.get(objectId));
}

function inspectColliderAt(clientX, clientY, { focus = true, spawnRobot = false } = {}) {
  if (!collisionTargets().length) return;
  setRaycasterFromCanvasPoint(clientX, clientY);
  const hits = intersectCollisionWorld(raycaster);
  if (!hits.length) {
    state.lastHit = "none";
    state.lastHitInfo = null;
    updateHud();
    showToast("No collider hit. Visual 3DGS splats are ignored by raycast.");
    return;
  }
  const hit = hits[0];
  const normal = worldHitNormal(hit);
  state.lastHitInfo = describeHit(hit, normal);
  state.lastHit = `${state.lastHitInfo.label}:face-${state.lastHitInfo.faceIndex}`;
  markHit(hit.point, normal);
  if (focus && state.cameraMode === "orbit") {
    setSmoothCameraTarget(hit.point, { preserveOffset: true, immediate: false });
  }
  if (spawnRobot) {
    respawnRobotAt(hit.point, "shift-click");
  }
  updateHud();
  const idText = state.lastHitInfo.objectId ?? "n/a";
  const probText = state.lastHitInfo.objectProbability ?? "n/a";
  showToast(spawnRobot
    ? `Robot respawned on collider face ${state.lastHitInfo.faceIndex}.`
    : `Collider hit face ${state.lastHitInfo.faceIndex}, object_id ${idText}, p=${probText}.`);
}

function onPointerDown(event) {
  if (![0, 1, 2].includes(event.button)) return;
  canvas.focus({ preventScroll: true });
  if (state.cameraMode === "orbit") {
    if (event.button === 0 && !event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) {
      const hit = interactiveObjectHitAt(event.clientX, event.clientY);
      if (hit && beginInteractiveObjectDrag(event, hit)) return;
    }
    event.preventDefault();
    state.robotFollowCamera = false;
    canvasDrag = {
      pointerId: event.pointerId,
      button: event.button,
      startX: event.clientX,
      startY: event.clientY,
      lastX: event.clientX,
      lastY: event.clientY,
      time: performance.now(),
      moved: false,
    };
    pointerDown = canvasDrag;
    canvas.setPointerCapture?.(event.pointerId);
    setLayerVisibility();
    return;
  }
  if (event.button !== 0) return;
  pointerDown = { x: event.clientX, y: event.clientY, time: performance.now(), moved: false };
}

function onPointerUp(event) {
  if (objectDrag && event.pointerId === objectDrag.pointerId) {
    event.preventDefault();
    endInteractiveObjectDrag(event);
    return;
  }
  if (state.cameraMode === "orbit" && canvasDrag && event.pointerId === canvasDrag.pointerId) {
    event.preventDefault();
    const dx = event.clientX - canvasDrag.startX;
    const dy = event.clientY - canvasDrag.startY;
    const elapsed = performance.now() - canvasDrag.time;
    const moved = canvasDrag.moved || Math.hypot(dx, dy) > 5;
    try {
      canvas.releasePointerCapture?.(canvasDrag.pointerId);
    } catch {
      // Pointer capture may already be released by the browser.
    }
    canvasDrag = null;
    pointerDown = null;
    if (!moved && elapsed <= 380 && event.button === 0) {
      inspectColliderAt(event.clientX, event.clientY, { spawnRobot: event.shiftKey });
    }
    return;
  }

  if (event.button !== 0 || !pointerDown) return;
  const dx = event.clientX - pointerDown.x;
  const dy = event.clientY - pointerDown.y;
  const elapsed = performance.now() - pointerDown.time;
  pointerDown = null;
  if (Math.hypot(dx, dy) > 5 || elapsed > 360) return;
  inspectColliderAt(event.clientX, event.clientY, { spawnRobot: event.shiftKey });
}

function endCanvasDrag(event) {
  if (objectDrag) endInteractiveObjectDrag(event, { cancelled: true });
  if (!canvasDrag) return;
  try {
    canvas.releasePointerCapture?.(canvasDrag.pointerId);
  } catch {
    // Pointer capture may already be released by the browser.
  }
  canvasDrag = null;
  pointerDown = null;
  event?.preventDefault?.();
}

function beginViewportGizmoDrag(event, mode) {
  event.preventDefault();
  event.stopPropagation();
  state.viewportGizmo = mode;
  state.robotFollowCamera = false;
  state.cameraMode = "orbit";
  gizmoDrag = {
    mode,
    pointerId: event.pointerId,
    element: event.currentTarget,
    lastX: event.clientX,
    lastY: event.clientY,
  };
  event.currentTarget.setPointerCapture?.(event.pointerId);
  event.currentTarget.classList.add("is-dragging");
  setLayerVisibility();
}

function updateViewportGizmoDrag(event) {
  if (!gizmoDrag) return;
  if (event.pointerId != null && gizmoDrag.pointerId != null && event.pointerId !== gizmoDrag.pointerId) return;
  event.preventDefault();
  const dx = event.clientX - gizmoDrag.lastX;
  const dy = event.clientY - gizmoDrag.lastY;
  if (dx === 0 && dy === 0) return;
  gizmoDrag.lastX = event.clientX;
  gizmoDrag.lastY = event.clientY;
  if (gizmoDrag.mode === "rotate") {
    orbitCameraAroundTarget({
      theta: -dx * CAMERA_ORBIT_SENSITIVITY,
      phi: -dy * CAMERA_ORBIT_SENSITIVITY,
    });
  } else {
    panCameraTarget(dx, dy);
  }
}

function endViewportGizmoDrag(event) {
  if (!gizmoDrag) return;
  event.preventDefault?.();
  const element = gizmoDrag.element;
  try {
    element?.releasePointerCapture?.(gizmoDrag.pointerId);
  } catch {
    // Pointer capture may already be released by the browser.
  }
  element?.classList.remove("is-dragging");
  state.viewportGizmo = "idle";
  gizmoDrag = null;
  setLayerVisibility();
}

function onCanvasWheel(event) {
  if (state.cameraMode !== "orbit") return;
  event.preventDefault();

  const now = performance.now();
  if (now - smoothCamera.lastWheelTime > 80) {
    smoothCamera.burstIsWheel = classifyWheelInput(event);
  }
  smoothCamera.lastWheelTime = now;

  const wheelDelta = event.shiftKey && event.deltaY === 0 ? event.deltaX : event.deltaY;
  const verticalDominant = Math.abs(event.deltaY) >= Math.abs(event.deltaX);
  if (event.shiftKey) {
    panCameraTarget(event.deltaX, event.deltaY);
  } else if (verticalDominant || smoothCamera.burstIsWheel || event.ctrlKey || event.metaKey) {
    zoomCameraByWheel(wheelDelta);
  } else {
    orbitCameraAroundTarget({
      theta: -event.deltaX * CAMERA_TRACKPAD_ORBIT_SENSITIVITY,
      phi: -event.deltaY * CAMERA_TRACKPAD_ORBIT_SENSITIVITY,
    });
  }
}

function updateFlyCamera(dt) {
  if (state.cameraMode !== "fly" || state.robotEnabled) return;
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  const flatForward = forward.clone().setY(0);
  if (flatForward.lengthSq() > 0) flatForward.normalize();
  const right = new THREE.Vector3().crossVectors(flatForward, camera.up).normalize();
  const move = new THREE.Vector3();
  if (keyState.has("KeyW") || keyState.has("ArrowUp")) move.add(flatForward);
  if (keyState.has("KeyS") || keyState.has("ArrowDown")) move.sub(flatForward);
  if (keyState.has("KeyA") || keyState.has("ArrowLeft")) move.sub(right);
  if (keyState.has("KeyD") || keyState.has("ArrowRight")) move.add(right);
  if (keyState.has("KeyE") || keyState.has("Space")) move.add(ROBOT_UP);
  if (keyState.has("KeyQ") || keyState.has("ShiftLeft") || keyState.has("ShiftRight")) move.sub(ROBOT_UP);
  if (move.lengthSq() === 0) return;
  const speed = keyState.has("AltLeft") || keyState.has("AltRight") ? 28 : 12;
  camera.position.addScaledVector(move.normalize(), speed * dt);
  controls.target.copy(camera.position).add(forward.multiplyScalar(5));
}

function onPointerMove(event) {
  if (updateInteractiveObjectDrag(event)) return;
  if (state.cameraMode === "orbit" && canvasDrag && event.pointerId === canvasDrag.pointerId) {
    event.preventDefault();
    const dx = event.clientX - canvasDrag.lastX;
    const dy = event.clientY - canvasDrag.lastY;
    if (dx === 0 && dy === 0) return;
    canvasDrag.lastX = event.clientX;
    canvasDrag.lastY = event.clientY;
    if (Math.hypot(event.clientX - canvasDrag.startX, event.clientY - canvasDrag.startY) > 4) {
      canvasDrag.moved = true;
    }

    const wantsZoom = event.ctrlKey || event.metaKey || canvasDrag.button === 1;
    const wantsPan = event.shiftKey || event.button === 2 || canvasDrag.button === 2;
    if (wantsZoom) {
      zoomCameraByWheel(-dy, CAMERA_DRAG_ZOOM_SENSITIVITY);
    } else if (wantsPan) {
      panCameraTarget(dx, dy);
    } else {
      orbitCameraAroundTarget({
        theta: -dx * CAMERA_ORBIT_SENSITIVITY,
        phi: -dy * CAMERA_ORBIT_SENSITIVITY,
      });
    }
    return;
  }
  if (state.cameraMode !== "fly" || event.buttons !== 1) return;
  const direction = new THREE.Vector3();
  camera.getWorldDirection(direction);
  const spherical = new THREE.Spherical().setFromVector3(direction);
  spherical.theta -= event.movementX * 0.0032;
  spherical.phi = THREE.MathUtils.clamp(spherical.phi - event.movementY * 0.0032, 0.04, Math.PI - 0.04);
  direction.setFromSpherical(spherical).normalize();
  controls.target.copy(camera.position).add(direction.multiplyScalar(5));
  camera.lookAt(controls.target);
}

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height, false);
  applyRendererPixelRatio(qualityPixelRatioLimit());
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function updateAlignmentButtons() {
  const offset = state.manualAlignmentOffset;
  const rotation = state.manualAlignmentRotationDeg;
  const defaultOffset = state.defaultViewerOffset;
  const label = document.querySelector("#alignmentReadout");
  if (label) {
    label.textContent = `default ${defaultOffset.x.toFixed(2)} ${defaultOffset.y.toFixed(2)} ${defaultOffset.z.toFixed(2)} · manual ${offset.x.toFixed(2)} ${offset.y.toFixed(2)} ${offset.z.toFixed(2)} · rot ${rotation.x.toFixed(1)}°/${rotation.y.toFixed(1)}°/${rotation.z.toFixed(1)}°`;
  }
  const reset = document.querySelector("#resetAlignment");
  if (reset) reset.classList.toggle("is-active", manualAlignmentIsActive());
}

function setLayerVisibility() {
  visualLayer.visible = state.showVisual;
  sparkRenderer.visible = state.showVisual && state.visualReady;
  if (visualSplat) {
    const staticVisualVisible = state.showVisual && state.showStaticVisual;
    visualSplat.visible = staticVisualVisible;
    visualSplat.opacity = staticVisualVisible ? 1 : 0;
  }
  applyInteractiveObjectVisibility();
  document.querySelector("#toggleVisual").classList.toggle("is-active", state.showVisual);
  const staticVisualButton = document.querySelector("#toggleStaticVisual");
  staticVisualButton.classList.toggle("is-active", state.showStaticVisual);
  staticVisualButton.textContent = state.showStaticVisual ? "Static Scene On" : "Static Scene Off";
  document.querySelector("#toggleCollider").classList.toggle("is-active", state.colliderRenderMode !== "hidden");
  document.querySelector("#toggleCollider").textContent = `Collider ${state.colliderRenderMode}`;
  const semanticButton = document.querySelector("#toggleSemantic");
  semanticButton.disabled = !state.colliderHasSemantics;
  semanticButton.classList.toggle("is-active", state.colliderHasSemantics && state.semanticColor);
  semanticButton.textContent = state.colliderHasSemantics
    ? (state.semanticColor ? "Semantic color" : "Raw mesh color")
    : "Raw mesh color";
  document.querySelector("#toggleCameraMode").classList.toggle("is-active", state.cameraMode === "fly");
  document.querySelector("#toggleCameraMode").textContent = state.cameraMode === "fly" ? "Fly Camera" : "Orbit Camera";
  document.querySelectorAll("[data-camera-preset]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.cameraPreset === state.cameraPreset);
  });
  document.querySelector("#toggleRobot").classList.toggle("is-active", state.robotEnabled);
  document.querySelector("#toggleRobotFollow").classList.toggle("is-active", state.robotFollowCamera);
  document.querySelector("#toggleRobotFollow").textContent = state.robotFollowCamera ? "Follow On" : "Follow Off";
  document.querySelector("#toggleRobotCollision").classList.toggle("is-active", state.robotObstacleCollision);
  document.querySelector("#toggleRobotCollision").textContent = state.robotObstacleCollision ? "Collision On" : "Collision Off";
  document.querySelector("#toggleQuality").classList.toggle("is-active", state.qualityMode === "performance");
  document.querySelector("#toggleQuality").textContent = state.qualityMode === "performance" ? "Performance" : "Balanced";
  const objectProxyButton = document.querySelector("#toggleObjectProxies");
  objectProxyButton.classList.toggle("is-active", state.interactiveObjectProxiesVisible);
  objectProxyButton.textContent = state.interactiveObjectProxiesVisible ? "Object Proxies On" : "Object Proxies Off";
  const objectSoloButton = document.querySelector("#toggleObjectSolo");
  objectSoloButton.classList.toggle("is-active", state.interactiveObjectSolo);
  objectSoloButton.textContent = state.interactiveObjectSolo ? "Solo Object On" : "Solo Object Off";
  controls.enabled = false;
  robotLayer.visible = state.robotEnabled;
  if (robotGroup) robotGroup.visible = state.robotEnabled;
  applyColliderMode();
  updateAlignmentButtons();
  updateHud();
}

function acceptsTextInput(target) {
  return target instanceof Element && Boolean(target.closest("input, textarea, select, [contenteditable='true']"));
}

function bindEvents() {
  window.addEventListener("resize", resize);
  window.addEventListener("pointermove", updateViewportGizmoDrag);
  window.addEventListener("pointerup", endViewportGizmoDrag);
  window.addEventListener("pointercancel", endViewportGizmoDrag);
  window.addEventListener("keydown", (event) => {
    if (acceptsTextInput(event.target)) return;
    const wasPressed = keyState.has(event.code);
    keyState.add(event.code);
    const robotKey = ["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.code);
    if (robotKey && !wasPressed) pulseRobotFromKey(event.code);
    if (event.code === "Space" && !wasPressed && state.robotEnabled) startRobotJump();
    if (robotKey || ["Space"].includes(event.code)) {
      event.preventDefault();
    }
  });
  window.addEventListener("keyup", (event) => keyState.delete(event.code));
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", endCanvasDrag);
  canvas.addEventListener("lostpointercapture", endCanvasDrag);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("dblclick", onCanvasDoubleClick);
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  canvas.addEventListener("wheel", onCanvasWheel, { passive: false });
  sceneQaForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    previewSceneInput(sceneQaInput?.value || "");
  });
  sceneCommandCancel?.addEventListener("click", () => {
    clearSceneCommandPreview();
    updateHud();
  });
  sceneCommandConfirm?.addEventListener("click", () => {
    submitCurrentSceneCommand();
  });
  document.querySelectorAll("[data-viewport-gizmo]").forEach((button) => {
    button.addEventListener("pointerdown", (event) => beginViewportGizmoDrag(event, button.dataset.viewportGizmo));
    button.addEventListener("lostpointercapture", () => {
      button.classList.remove("is-dragging");
      state.viewportGizmo = "idle";
      gizmoDrag = null;
      setLayerVisibility();
    });
  });

  document.querySelector("#toggleVisual").addEventListener("click", () => {
    state.showVisual = !state.showVisual;
    setLayerVisibility();
    if (state.showVisual && (!state.visualReady || state.interactiveObjectReadyCount < state.interactiveObjectCount)) {
      const loadVisuals = EXPLICIT_MANIFEST_URL == null
        ? ensureInteractiveObjectsLoaded()
        : ensureVisualSplatLoaded();
      loadVisuals.catch((error) => {
        console.error(error);
        state.error = error?.message || String(error);
        showToast(`Visual load failed: ${state.error}`);
        syncDebugState();
      });
    }
  });
  document.querySelector("#toggleStaticVisual").addEventListener("click", () => {
    state.showStaticVisual = !state.showStaticVisual;
    setLayerVisibility();
    showToast(state.showStaticVisual ? "Static Gaussian scene visible." : "Showing interactive Gaussian objects only.");
  });
  document.querySelector("#toggleCollider").addEventListener("click", () => {
    state.colliderRenderMode = nextValue(COLLIDER_MODES, state.colliderRenderMode);
    setLayerVisibility();
    showToast(`Collider render mode: ${COLLIDER_LABELS[state.colliderRenderMode]}. Raycast stays active.`);
  });
  document.querySelector("#toggleSemantic").addEventListener("click", () => {
    if (!state.colliderHasSemantics) {
      showToast("This TSDF collider has no semantic attributes.");
      return;
    }
    state.semanticColor = !state.semanticColor;
    applySemanticColorMode();
    setLayerVisibility();
    showToast(state.semanticColor ? "Semantic object colors enabled." : "Raw mesh vertex colors enabled.");
  });
  document.querySelector("#toggleCameraMode").addEventListener("click", () => {
    state.cameraMode = state.cameraMode === "orbit" ? "fly" : "orbit";
    state.cameraPreset = state.cameraMode === "fly" ? "fly" : "custom";
    state.cameraPresetSource = state.cameraMode === "fly" ? "free-fly" : "free-orbit";
    keyState.clear();
    if (state.cameraMode === "orbit") syncSmoothCameraFromCurrent();
    setLayerVisibility();
    showToast(state.cameraMode === "fly"
      ? "Fly camera enabled. Turn Robot off if you want WASD to move the camera."
      : "Orbit camera enabled.");
  });
  document.querySelector("#toggleRobot").addEventListener("click", () => {
    state.robotEnabled = !state.robotEnabled;
    setLayerVisibility();
    showToast(state.robotEnabled ? "Robot enabled on mesh collider." : "Robot hidden; collider raycast stays active.");
  });
  document.querySelector("#toggleRobotFollow").addEventListener("click", () => {
    state.robotFollowCamera = !state.robotFollowCamera;
    if (state.robotFollowCamera && robotGroup) updateRobotFollowCamera(0, true);
    setLayerVisibility();
    showToast(state.robotFollowCamera ? "Camera follows robot." : "Camera follow disabled.");
  });
  document.querySelector("#toggleRobotCollision").addEventListener("click", () => {
    state.robotObstacleCollision = !state.robotObstacleCollision;
    state.robotBlocked = false;
    setLayerVisibility();
    showToast(state.robotObstacleCollision ? "Robot obstacle rays enabled." : "Robot uses ground-only movement.");
  });
  document.querySelector("#toggleQuality").addEventListener("click", () => {
    state.qualityMode = nextValue(QUALITY_MODES, state.qualityMode);
    applyQualityMode({ announce: true });
    setLayerVisibility();
  });
  document.querySelector("#toggleObjectProxies").addEventListener("click", () => {
    state.interactiveObjectProxiesVisible = !state.interactiveObjectProxiesVisible;
    setLayerVisibility();
    showToast(state.interactiveObjectProxiesVisible
      ? "Interactive object mesh proxies visible."
      : "Interactive object mesh proxies hidden but still selectable.");
  });
  document.querySelector("#focusObject").addEventListener("click", focusNextInteractiveObject);
  document.querySelector("#toggleObjectSolo").addEventListener("click", () => {
    state.interactiveObjectSolo = !state.interactiveObjectSolo;
    if (state.interactiveObjectSolo && !state.selectedInteractiveObject) focusNextInteractiveObject();
    setLayerVisibility();
    showToast(state.interactiveObjectSolo ? "Only the selected interactive object is visible." : "All interactive objects are visible.");
  });
  document.querySelector("#interiorView").addEventListener("click", () => {
    frameCameraToInterior({ announce: true });
  });
  document.querySelectorAll("[data-camera-preset]").forEach((button) => {
    button.addEventListener("click", () => {
      setCameraOrbitView(button.dataset.cameraPreset, { announce: true });
    });
  });
  document.querySelectorAll("[data-camera-orbit]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.cameraOrbit;
      const turn = THREE.MathUtils.degToRad(18);
      const tilt = THREE.MathUtils.degToRad(12);
      if (action === "left") orbitCameraAroundTarget({ theta: -turn, announce: true });
      if (action === "right") orbitCameraAroundTarget({ theta: turn, announce: true });
      if (action === "up") orbitCameraAroundTarget({ phi: -tilt, announce: true });
      if (action === "down") orbitCameraAroundTarget({ phi: tilt, announce: true });
      if (action === "in") orbitCameraAroundTarget({ scale: 0.78, announce: true });
      if (action === "out") orbitCameraAroundTarget({ scale: 1.28, announce: true });
      const travelStep = Math.max(0.8, Math.min(2.4, smoothCamera.desiredSpherical.radius * 0.22));
      if (action === "advance") moveCameraRigAlongView(travelStep, { announce: true });
      if (action === "retreat") moveCameraRigAlongView(-travelStep, { announce: true });
    });
  });
  document.querySelector("#respawnRobot").addEventListener("click", () => {
    spawnRobotAtBounds();
    setLayerVisibility();
    showToast("Robot returned to the fixed interior spawn.");
  });
  document.querySelector("#jumpRobot").addEventListener("click", startRobotJump);
  document.querySelector("#resetView").addEventListener("click", () => {
    const initialPreset = manifest?.initialState?.cameraPreset || "reference";
    setCameraOrbitView(initialPreset, { announce: true, immediate: true });
  });
  document.querySelectorAll("[data-robot-step]").forEach((button) => {
    button.addEventListener("click", () => stepRobotByCamera(button.dataset.robotStep));
  });
  document.querySelectorAll("[data-align-nudge]").forEach((button) => {
    button.addEventListener("click", () => {
      const [axis, direction] = button.dataset.alignNudge.split(":");
      const delta = new THREE.Vector3();
      delta[axis] = Number(direction) * ALIGNMENT_NUDGE_STEP;
      nudgeVisualAlignment(delta);
    });
  });
  document.querySelectorAll("[data-align-rotate]").forEach((button) => {
    button.addEventListener("click", () => {
      const [axis, direction] = button.dataset.alignRotate.split(":");
      nudgeVisualRotation(axis, Number(direction) * ALIGNMENT_ROTATION_STEP_DEG);
    });
  });
  document.querySelector("#resetAlignment").addEventListener("click", resetManualAlignment);
  exposeDebugApi();
}

async function init() {
  try {
    bindEvents();
    resize();
    updateHud();
    await loadManifest();
    loadManualAlignment();
    visualMetaBox = boxFromMeta(manifest.assets.visual);
    const referenceBoundsAssetKey = manifest.presentation?.referenceBoundsAssetKey || "collider";
    colliderMetaBox = boxFromMeta(manifest.assets[referenceBoundsAssetKey] || manifest.assets.collider);
    baseVisualTransform = calculateVisualToColliderTransform(visualMetaBox, colliderMetaBox, manifest.alignment);
    await loadColliderMesh();
    initializeInteractiveObjectComponents();
    applyEffectiveVisualTransform();
    await ensureInteractiveObjectCollidersLoaded();
    transformedVisualBounds = calculateTransformedBox(visualMetaBox, buildEffectiveVisualTransform());
    spawnRobotAtBounds();
    frameCameraToInterior();
    applySemanticColorMode();
    applyQualityMode();
    setLayerVisibility();
    if (START_WITH_STATIC_VISUAL) await ensureVisualSplatLoaded();
    else if (state.showVisual) await ensureInteractiveObjectsLoaded();
    applyQualityMode();
    setLayerVisibility();
    syncDebugState();
  } catch (error) {
    console.error(error);
    state.error = error?.message || String(error);
    modeChip.textContent = `Load failed: ${state.error}`;
    showToast(`Load failed: ${state.error}`);
    syncDebugState();
  }
}

function animate() {
  const frameDelta = Math.min(clock.getDelta(), PHYSICS_MAX_FRAME_DELTA);
  const cameraDelta = Math.min(frameDelta, 0.05);
  updateFlyCamera(cameraDelta);
  let physicsRemaining = frameDelta;
  let physicsSubsteps = 0;
  while (physicsRemaining > 1e-6) {
    const physicsDelta = Math.min(PHYSICS_FIXED_STEP, physicsRemaining);
    updateRobot(physicsDelta);
    updateInteractiveObjectAnimations(physicsDelta);
    physicsRemaining -= physicsDelta;
    physicsSubsteps += 1;
  }
  state.physicsFrameDelta = Number(frameDelta.toFixed(4));
  state.physicsSubsteps = physicsSubsteps;
  updateSmoothCamera(cameraDelta);
  if (state.visualReady) sparkRenderer.render(scene, camera);
  else renderer.render(scene, camera);

  frameCount += 1;
  const now = performance.now();
  if (now - lastFpsUpdate > 500) {
    state.fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate || 1));
    fpsMetric.textContent = state.fps.toString();
    updateAdaptiveQuality(state.fps);
    updateHud();
    frameCount = 0;
    lastFpsUpdate = now;
  }
  requestAnimationFrame(animate);
}

lastFpsUpdate = performance.now();
init();
animate();
