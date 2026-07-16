# video2world

video2world turns a calibrated scene video into a layered, queryable 3D world:

```text
video
  -> frames, cameras, DA3 depth and point prior
  -> PGSR scene Gaussian PLY and TSDF scene mesh
  -> open-vocabulary discovery and SAM3 masks
  -> multi-view 2D-to-3D instance fusion and semantic Gaussian PLY
  -> detailed object descriptions and scene relations
  -> EmbodiedGen V2/TRELLIS object Gaussian + mesh candidates
  -> scene-space placement, collision proxies and QA gates
  -> interactive Web world with robot navigation and scene cognition QA
```

The project is deliberately independent from Video2Mesh, Holi-Spatial,
EmbodiedGen and the relumeow publishing repository. Those systems remain
upstream providers behind explicit adapters; video2world owns orchestration,
cross-stage contracts, asset validation, world assembly and the runtime viewer.

Current design evidence is documented under [docs/video2world/](docs/video2world/README.md).
Generated models and run outputs are intentionally excluded from Git.

## What is implemented

- Content-addressed ten-stage orchestration with `init`, `plan`, `run`, and
  `adopt-existing` workflows.
- A versioned, shell-free external-provider contract and a complete
  Holi-Spatial/Video2Mesh/EmbodiedGen profile. Site Python interpreters,
  repositories, checkpoints, and driver argv are checked before a model starts;
  outputs are role-checked, hashed, and recorded in redacted receipts.
- A strict `world-manifest-1.0.0` model with local hash validation and
  fail-closed scene queries.
- A versioned Web deployment manifest validated before the runtime loads any
  scene or object assets.
- Spark/Three.js layered rendering, GLB/BVH collision, robot navigation,
  object focus, drag rotation, double-click 360-degree rotation, and bilingual
  location/appearance queries.
- A verified Bedroom 4 adoption example: 1,512,870 visual primitives, four GLB
  colliders, one visual-only pillow component, and desktop/mobile browser QA.

The checked-in root demo is intentionally a tiny fixture. Real PGSR, TSDF,
semantic Gaussian, TRELLIS, and cognition assets remain local or remote and are
referenced by hash-bearing manifests.

The provider profile is an executable integration boundary, not a claim that
arbitrary videos are zero-configuration or already verified. Copy
`video2world/configs/holi_embodiedgen.provider.example.yaml`, bind its site
drivers and model environments, then initialize with
`video2world/configs/holi_embodiedgen_upstream.yaml`. Bedroom 4 remains the
separate scene-specific verified adoption sample.

## Quick start

```bash
uv sync --all-groups
npm install

uv run pytest
uv run ruff check .
npm test
npm run test:e2e
npm run dev -- --port 4175
```

Open `http://127.0.0.1:4175/` for the committed fixture. A materialized world is
selected explicitly, for example:

```text
http://127.0.0.1:4175/?manifest=/worlds/bedroom4/manifest.json
```

See the [pipeline guide](docs/video2world/project-docs/pipeline.md),
[manifest contract](docs/video2world/project-docs/world-manifest.md), and
[Bedroom 4 evidence report](docs/video2world/progress/bedroom4-20260716.md).
