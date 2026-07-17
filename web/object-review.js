import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const VIEW_SPECS = [
  {
    id: "front",
    direction: new THREE.Vector3(0, 0, 1),
    up: new THREE.Vector3(0, 1, 0),
  },
  {
    id: "right",
    direction: new THREE.Vector3(1, 0, 0),
    up: new THREE.Vector3(0, 1, 0),
  },
  {
    id: "back",
    direction: new THREE.Vector3(0, 0, -1),
    up: new THREE.Vector3(0, 1, 0),
  },
  {
    id: "left",
    direction: new THREE.Vector3(-1, 0, 0),
    up: new THREE.Vector3(0, 1, 0),
  },
  {
    id: "top",
    direction: new THREE.Vector3(0, 1, 0),
    up: new THREE.Vector3(0, 0, -1),
  },
  {
    id: "bottom",
    direction: new THREE.Vector3(0, -1, 0),
    up: new THREE.Vector3(0, 0, 1),
  },
];

const params = new URLSearchParams(window.location.search);
const assetUrl = params.get("asset");
const objectId = params.get("objectId") || "unnamed-object";
const title = document.querySelector("#object-title");
const grid = document.querySelector("#review-grid");
const status = document.querySelector("#review-status");
const stats = document.querySelector("#review-stats");

const reviewState = {
  state: "loading",
  objectId,
  assetUrl,
  renderMode: "neutral-albedo",
  views: VIEW_SPECS.map(({ id }) => id),
  viewSpecs: VIEW_SPECS.map(({ id, direction, up }) => ({
    id,
    direction: direction.toArray(),
    up: up.toArray(),
  })),
  bounds: null,
  meshCount: 0,
  triangleCount: 0,
  renderDataUrls: {},
  error: null,
};
window.__VIDEO2WORLD_OBJECT_REVIEW__ = reviewState;

title.textContent = objectId;

function formatVector(values) {
  return values.map((value) => value.toFixed(4)).join(" x ");
}

function addStat(label, value) {
  const item = document.createElement("div");
  item.className = "review-stat";
  const term = document.createElement("dt");
  term.textContent = label;
  const description = document.createElement("dd");
  description.textContent = value;
  item.append(term, description);
  stats.append(item);
}

function inspectGeometry(root) {
  let meshCount = 0;
  let triangleCount = 0;
  root.traverse((node) => {
    if (!node.isMesh || !node.geometry) return;
    meshCount += 1;
    const indexCount = node.geometry.index?.count;
    const positionCount = node.geometry.attributes.position?.count || 0;
    triangleCount += Math.floor((indexCount || positionCount) / 3);
    node.castShadow = true;
    node.receiveShadow = true;
  });
  return { meshCount, triangleCount };
}

function neutralAlbedoClone(root) {
  const clone = root.clone(true);
  clone.traverse((node) => {
    if (!node.isMesh) return;
    const sourceMaterials = Array.isArray(node.material) ? node.material : [node.material];
    const materials = sourceMaterials.map((source) => {
      const material = new THREE.MeshBasicMaterial({
        color: source?.color?.clone?.() || new THREE.Color(0xffffff),
        map: source?.map || null,
        alphaMap: source?.alphaMap || null,
        alphaTest: Number(source?.alphaTest || 0),
        opacity: Number(source?.opacity ?? 1),
        transparent: Boolean(source?.transparent || Number(source?.opacity ?? 1) < 1),
        side: THREE.DoubleSide,
        vertexColors: Boolean(source?.vertexColors),
      });
      material.name = `${source?.name || "material"} neutral-albedo-review`;
      return material;
    });
    node.material = Array.isArray(node.material) ? materials : materials[0];
    node.castShadow = false;
    node.receiveShadow = false;
  });
  return clone;
}

function makeView(root, box, sphere, spec) {
  const panel = document.createElement("article");
  panel.className = "review-view";
  panel.dataset.viewId = spec.id;
  const label = document.createElement("p");
  label.className = "review-label";
  label.textContent = spec.id;
  panel.append(label);
  grid.append(panel);

  const scene = new THREE.Scene();
  scene.add(neutralAlbedoClone(root));

  const edgeGeometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(
    box.max.x - box.min.x,
    box.max.y - box.min.y,
    box.max.z - box.min.z,
  ));
  const edge = new THREE.LineSegments(
    edgeGeometry,
    new THREE.LineBasicMaterial({ color: 0xa66b22, transparent: true, opacity: 0.75 }),
  );
  edge.position.copy(sphere.center);
  const width = Math.max(panel.clientWidth, 320);
  const height = Math.max(panel.clientHeight, 240);
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(width, height, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  panel.append(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(30, width / height, 0.001, 1000);
  const distance = Math.max(sphere.radius / Math.sin(THREE.MathUtils.degToRad(15)), 0.25) * 1.18;
  camera.position.copy(sphere.center).addScaledVector(spec.direction, distance);
  camera.up.copy(spec.up);
  camera.lookAt(sphere.center);
  camera.near = Math.max(distance / 100, 0.001);
  camera.far = distance * 8;
  camera.updateProjectionMatrix();
  renderer.setClearColor(0x000000, 0);
  renderer.render(scene, camera);
  const objectOnlyDataUrl = renderer.domElement.toDataURL("image/png");
  scene.add(edge);
  renderer.render(scene, camera);

  return { renderer, camera, scene, panel, objectOnlyDataUrl };
}

async function loadReview() {
  if (!assetUrl) throw new Error("Missing required ?asset= URL parameter");
  const gltf = await new GLTFLoader().loadAsync(assetUrl);
  const root = gltf.scene;
  root.updateWorldMatrix(true, true);
  const box = new THREE.Box3().setFromObject(root);
  if (box.isEmpty()) throw new Error("Loaded asset has no renderable bounds");
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const size = box.getSize(new THREE.Vector3());
  const geometry = inspectGeometry(root);
  if (geometry.meshCount === 0 || geometry.triangleCount === 0) {
    throw new Error("Loaded asset has no triangle geometry");
  }

  reviewState.bounds = {
    min: box.min.toArray(),
    max: box.max.toArray(),
    size: size.toArray(),
    radius: sphere.radius,
  };
  reviewState.meshCount = geometry.meshCount;
  reviewState.triangleCount = geometry.triangleCount;

  addStat("Extents", formatVector(reviewState.bounds.size));
  addStat("Meshes", geometry.meshCount.toLocaleString("en-US"));
  addStat("Triangles", geometry.triangleCount.toLocaleString("en-US"));
  addStat("Radius", sphere.radius.toFixed(4));
  VIEW_SPECS.forEach((spec) => {
    const view = makeView(root, box, sphere, spec);
    reviewState.renderDataUrls[spec.id] = view.objectOnlyDataUrl;
    const artifact = document.createElement("img");
    artifact.className = "review-artifact";
    artifact.dataset.reviewRender = spec.id;
    artifact.alt = `${spec.id} object-only render`;
    artifact.src = view.objectOnlyDataUrl;
    document.body.append(artifact);
  });
  reviewState.state = "ready";
  status.textContent = "Ready for deterministic six-view review.";
  status.dataset.state = "ready";
}

loadReview().catch((error) => {
  reviewState.state = "error";
  reviewState.error = error instanceof Error ? error.message : String(error);
  status.textContent = `Review failed: ${reviewState.error}`;
  status.dataset.state = "error";
  console.error(error);
});
