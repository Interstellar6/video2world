#!/usr/bin/env python3
"""Generate the delivery web viewer next to an export.

The old delivery shipped a small three.js app that lets someone pick an object
and look at its mesh and its Gaussian splat without opening a desktop tool. This
regenerates that app for whatever the export actually contains: the page reads a
generated ``assets.json``, so nothing about the scene or the object list is
hardcoded, and every path is relative to the export root.

The generated folder is a normal Vite project (``npm install && npm run build``)
and the build output is self-contained, so it can be served with any static file
server rooted at the export directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.engine import PipelineError, read_json, sha256, write_json  # noqa: E402

INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>video2world delivery viewer</title>
    <link rel="stylesheet" href="./styles.css" />
  </head>
  <body>
    <canvas id="scene"></canvas>
    <aside id="panel">
      <h1 id="title">delivery</h1>
      <p class="metric">scene: <span id="sceneMetric">loading</span></p>
      <p class="metric">objects: <span id="objectMetric">loading</span></p>
      <div class="row">
        <button id="fitAll" type="button">fit</button>
        <button id="toggleScene" type="button">scene</button>
        <button id="toggleObjects" type="button">objects</button>
        <button id="toggleBg" type="button">background</button>
      </div>
      <ul id="objectList"></ul>
      <p class="hint" id="status">serve the export root and open /web-demo/dist/index.html</p>
    </aside>
    <script type="module" src="./app.js"></script>
  </body>
</html>
"""

STYLES = """:root { color-scheme: dark; }
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: #05070a; color: #e8eef5; font: 13px/1.5 system-ui, sans-serif; }
#scene { position: fixed; inset: 0; width: 100%; height: 100%; display: block; }
#panel { position: fixed; top: 12px; left: 12px; width: 260px; padding: 12px 14px; border-radius: 10px;
         background: rgba(10, 14, 20, 0.82); backdrop-filter: blur(6px); border: 1px solid rgba(255, 255, 255, 0.08); }
#panel h1 { margin: 0 0 8px; font-size: 14px; letter-spacing: 0.02em; }
.metric { margin: 2px 0; color: #9fb0c3; }
.row { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
button { flex: 1 1 auto; padding: 5px 8px; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.16);
         background: rgba(255, 255, 255, 0.06); color: inherit; cursor: pointer; }
button:hover { background: rgba(255, 255, 255, 0.14); }
#objectList { list-style: none; margin: 8px 0 0; padding: 0; max-height: 45vh; overflow: auto; }
#objectList li { padding: 4px 6px; border-radius: 6px; cursor: pointer; color: #c7d4e2; }
#objectList li.active, #objectList li:hover { background: rgba(90, 170, 255, 0.18); }
.hint { color: #7f8fa1; margin: 10px 0 0; }
"""

APP_JS = """import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

const params = new URLSearchParams(window.location.search);
let assetRoot = params.get("root") ?? "";
while (assetRoot.endsWith("/")) assetRoot = assetRoot.slice(0, -1);
const manifest = await (await fetch("./assets.json")).json();
for (const entry of manifest.objects) {
  for (const key of ["mesh", "splat"]) if (entry[key]) entry[key] = assetRoot + entry[key];
}
for (const entry of manifest.scene) entry.url = assetRoot + entry.url;
if (manifest.background) manifest.background = assetRoot + manifest.background;
const canvas = document.querySelector("#scene");
const status = document.querySelector("#status");
document.querySelector("#title").textContent = manifest.title || "delivery";

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070a);
scene.add(new THREE.HemisphereLight(0xffffff, 0x223244, 2.2));
const key = new THREE.DirectionalLight(0xffffff, 2.4);
key.position.set(6, 12, 8);
scene.add(key);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 500);
camera.position.set(4, 3, 4);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const sceneGroup = new THREE.Group();
const objectGroup = new THREE.Group();
const backgroundGroup = new THREE.Group();
scene.add(sceneGroup, objectGroup, backgroundGroup);

const gltfLoader = new GLTFLoader();
const plyLoader = new PLYLoader();

function decodedSplatColors(geometry) {
  // EmbodiedGen and PGSR store the DC term of the spherical harmonics, so the
  // rendered colour is 0.5 + SH_C0 * dc with SH_C0 = 0.28209479177387814.
  const dc = ["f_dc_0", "f_dc_1", "f_dc_2"].map((name) => geometry.getAttribute(name));
  if (dc.some((attribute) => !attribute)) return false;
  const count = geometry.getAttribute("position").count;
  const colors = new Float32Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      colors[index * 3 + channel] = Math.min(1, Math.max(0, 0.5 + 0.28209479177387814 * dc[channel].getX(index)));
    }
  }
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return true;
}

function pointsFromGeometry(geometry, size) {
  const hasColour = decodedSplatColors(geometry) || Boolean(geometry.getAttribute("color"));
  return new THREE.Points(geometry, new THREE.PointsMaterial({
    size, sizeAttenuation: true, vertexColors: hasColour, color: hasColour ? 0xffffff : 0xb9c6d4,
  }));
}

async function loadPly(url, size) {
  const geometry = await plyLoader.loadAsync(url);
  return pointsFromGeometry(geometry, size);
}

async function loadGltf(url) {
  const gltf = await gltfLoader.loadAsync(url);
  gltf.scene.traverse((node) => {
    if (node.isMesh) node.material = new THREE.MeshStandardMaterial({
      color: 0xffffff, vertexColors: Boolean(node.geometry.getAttribute("color")),
      roughness: 0.85, metalness: 0.0, side: THREE.DoubleSide,
    });
  });
  return gltf.scene;
}

let loaded = 0;
let failed = 0;
function report() {
  document.querySelector("#objectMetric").textContent = `${loaded}/${manifest.objects.length} loaded${failed ? ` · ${failed} failed` : ""}`;
}

for (const entry of manifest.scene || []) {
  loadPly(entry.url, entry.size || 0.012)
    .then((object) => { sceneGroup.add(object); document.querySelector("#sceneMetric").textContent = "loaded"; })
    .catch(() => { document.querySelector("#sceneMetric").textContent = "unavailable"; });
}
if (manifest.background) {
  loadGltf(manifest.background).then((node) => backgroundGroup.add(node)).catch(() => {});
}

const list = document.querySelector("#objectList");
for (const entry of manifest.objects) {
  const item = document.createElement("li");
  item.textContent = entry.label || entry.id;
  item.addEventListener("click", async () => {
    for (const node of [...objectGroup.children]) objectGroup.remove(node);
    [...list.children].forEach((child) => child.classList.remove("active"));
    item.classList.add("active");
    status.textContent = `loading ${entry.id}…`;
    let ok = 0;
    if (entry.mesh) {
      try { objectGroup.add(await loadGltf(entry.mesh)); ok += 1; } catch (error) { status.textContent = String(error); }
    }
    if (entry.splat) {
      try { objectGroup.add(await loadPly(entry.splat, entry.splatSize || 0.006)); ok += 1; } catch (error) { status.textContent = String(error); }
    }
    status.textContent = `${entry.id}: ${ok} representation(s) loaded`;
  });
  list.appendChild(item);
}
report();

document.querySelector("#fitAll").addEventListener("click", () => fit(new THREE.Box3().setFromObject(scene)));
document.querySelector("#toggleScene").addEventListener("click", () => { sceneGroup.visible = !sceneGroup.visible; });
document.querySelector("#toggleObjects").addEventListener("click", () => { objectGroup.visible = !objectGroup.visible; });
document.querySelector("#toggleBg").addEventListener("click", () => { backgroundGroup.visible = !backgroundGroup.visible; });

function fit(box) {
  if (box.isEmpty()) return;
  const centre = box.getCenter(new THREE.Vector3());
  const radius = box.getSize(new THREE.Vector3()).length() / 2 || 1;
  const distance = radius / Math.tan((camera.fov * Math.PI) / 360) * 1.4;
  camera.position.copy(centre).add(new THREE.Vector3(0.8, 0.6, 1).normalize().multiplyScalar(distance));
  camera.near = Math.max(0.001, radius / 200);
  camera.far = radius * 200;
  camera.updateProjectionMatrix();
  controls.target.copy(centre);
  controls.update();
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

renderer.render(scene, camera);
"""

PACKAGE_JSON = {
    "name": "video2world-delivery-viewer",
    "version": "0.1.0",
    "private": True,
    "type": "module",
    "scripts": {"dev": "vite", "build": "vite build", "preview": "vite preview"},
    "dependencies": {"three": "0.180.0"},
    "devDependencies": {"vite": "7.3.6"},
}

VITE_CONFIG = """import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  build: { outDir: "dist", emptyOutDir: true },
});
"""

README = """# Delivery viewer

```bash
npm install
npm run build          # writes dist/
python3 -m http.server 8000   # run this inside the export root
# then open http://127.0.0.1:8000/web-demo/dist/index.html
```

Serve the **export root**, not the viewer directory: asset URLs are absolute from
the server root, and `?root=<prefix>` overrides that when the viewer is hosted
somewhere else. `assets.json` is generated from the export manifest, so the scene
layers and the object list always match what was actually delivered. Meshes load as glTF and
Gaussian splats load as coloured points (the DC term of the spherical harmonics
is decoded in the browser); this is a QA viewer, not a photoreal splat renderer.
"""


def viewer_assets(export_root: Path, title: str) -> dict:
    manifest_path = export_root / "export_manifest.json"
    if not manifest_path.is_file():
        raise PipelineError(f"export manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)

    def url(relative: str) -> str:
        return "/" + relative.lstrip("/")

    scene = []
    for relative in sorted(manifest.get("scene_roles") or {}):
        if relative.endswith(".ply") and "depth" not in relative:
            scene.append({"role": relative, "url": url(relative),
                          "size": 0.02 if "3dgs" in relative else 0.01})
    objects = []
    version = manifest.get("object_version_2") or {}
    for object_id in sorted(version):
        mesh = export_root / "object" / "object_version_2" / object_id / "asset_mesh.glb"
        splat = export_root / "object" / "object_version_2" / object_id / "asset_splat.ply"
        entry = {"id": object_id, "label": object_id.replace("_", " ")}
        if mesh.is_file():
            entry["mesh"] = url(f"object/object_version_2/{object_id}/asset_mesh.glb")
        if splat.is_file():
            entry["splat"] = url(f"object/object_version_2/{object_id}/asset_splat.ply")
        if "mesh" in entry or "splat" in entry:
            objects.append(entry)
    for object_id in sorted(manifest.get("object_version_1") or {}):
        if any(entry["id"] == object_id for entry in objects):
            continue
        mesh = export_root / "object" / "object_version_1" / object_id / f"{object_id}.glb"
        if mesh.is_file():
            objects.append({"id": object_id, "label": object_id.replace("_", " "),
                            "mesh": url(f"object/object_version_1/{object_id}/{object_id}.glb")})
    background = export_root / "object" / "background.glb"
    return {"title": title, "scene": scene, "objects": objects,
            "background": url("object/background.glb") if background.is_file() else None}


def run(export_root: Path, *, title: str = "video2world delivery", build: bool = False) -> dict:
    export_root = export_root.resolve()
    payload = viewer_assets(export_root, title)
    directory = export_root / "web-demo"
    directory.mkdir(parents=True, exist_ok=True)
    files = {"index.html": INDEX_HTML, "styles.css": STYLES, "app.js": APP_JS,
             "vite.config.js": VITE_CONFIG, "README.md": README,
             "package.json": json.dumps(PACKAGE_JSON, indent=2, sort_keys=True) + "\n"}
    written = {}
    for name, content in files.items():
        path = directory / name
        path.write_text(content)
        written[name] = sha256(path)
    # Vite copies public/ into the build root, so ./assets.json resolves both in
    # the dev server and in dist/ without any base-path arithmetic.
    public = directory / "public"
    public.mkdir(parents=True, exist_ok=True)
    write_json(public / "assets.json", payload)
    written["public/assets.json"] = sha256(public / "assets.json")
    report = {"schema_version": "1.0", "kind": "video2world-modeling.delivery_viewer",
              "status": "generated", "directory": "web-demo", "title": title,
              "scene_layers": [entry["role"] for entry in payload["scene"]],
              "objects": [entry["id"] for entry in payload["objects"]],
              "files": written, "build_command": "cd web-demo && npm install && npm run build"}
    write_json(directory / "viewer.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--title", default="video2world delivery")
    args = parser.parse_args(argv)
    try:
        report = run(args.export_root, title=args.title)
    except (PipelineError, OSError, ValueError, KeyError) as error:
        print(f"build_viewer failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
