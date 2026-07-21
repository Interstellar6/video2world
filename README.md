# video2world

video2world turns a calibrated scene video into a layered, queryable 3D world:

```text
video
  -> frames, cameras, DA3 depth and point prior
  -> PGSR scene Gaussian PLY and TSDF scene mesh
  -> open-vocabulary discovery and SAM3 masks
  -> multi-view 2D-to-3D instance fusion and semantic Gaussian PLY
  -> detailed object descriptions and scene relations
  -> EmbodiedGen V2/TRELLIS mesh-first PBR GLB candidates
  -> scene-space placement, unified GLB surface collision and QA gates
  -> interactive Web world with robot navigation and scene cognition QA
```

The project is deliberately independent from Video2Mesh, Holi-Spatial,
EmbodiedGen and the relumeow publishing repository. Those systems remain
upstream providers behind explicit adapters; video2world owns orchestration,
cross-stage contracts, asset validation, world assembly and the runtime viewer.

Current design evidence is documented under [docs/video2world/](docs/video2world/README.md).
Generated models and run outputs are intentionally excluded from Git.

## What is implemented

- Content-addressed twelve-stage orchestration with `init`, `plan`, `run`, and
  `adopt-existing` workflows.
- A versioned, shell-free external-provider contract and a complete
  Holi-Spatial/Video2Mesh/EmbodiedGen profile. Site Python interpreters,
  repositories, checkpoints, and driver argv are checked before a model starts;
  outputs are role-checked, hashed, and recorded in redacted receipts.
- A strict `world-manifest-1.0.0` model with local hash validation and
  fail-closed scene queries.
- A versioned Web deployment manifest validated before the runtime loads any
  scene or object assets.
- Spark/Three.js layered rendering, mesh-first unified PBR GLB objects,
  GLB/MeshBVH character surface collision, robot navigation, object focus,
  drag rotation, double-click 360-degree rotation, and bilingual
  location/appearance queries. Per-object Gaussian/point cloud is optional.
- A verified Bedroom 4 legacy-adoption example: 1,512,870 visual primitives,
  four separate GLB colliders, one visual-only pillow component, and
  desktop/mobile browser QA. This historical production mode is retained while
  newer one-PBR-GLB object candidates are gated separately.

The checked-in root demo is intentionally a tiny fixture. Real PGSR, TSDF,
semantic Gaussian, TRELLIS, and cognition assets remain local or remote and are
referenced by hash-bearing manifests.

The current Bedroom 4 mesh-first candidate is a TRELLIS2 PBR GLB with 60,237
vertices and 97,082 faces. It is finite, nondegenerate, winding-consistent, and
non-watertight, so its honest collision contract is `surface_bvh`: character
surface blocking without volume or inside/outside claims. The direct local
three-pillow candidate now has a passed no-clean-plate browser QA report at
`examples/bedroom4/completion/direct-trellis2-refit/candidate/browser-qa-report.json`;
it verifies desktop/mobile unified GLB loading, MeshBVH robot blocking, pointer
focus, yaw transforms and the logical-only bed ancestor. The older verified
production bundle remains a separate historical baseline.

The provider profile is an executable integration boundary, not a claim that
arbitrary videos are zero-configuration or already verified. Copy
`video2world/configs/site_profile.example.yaml` and
`video2world/configs/site_provider.example.yaml`, then use `site-init` with
explicit named checkout and artifact roots. `site-preflight` launches no models;
`site-run` records each stage as executed or adopted and fails closed when a
provider is absent. The older `holi_embodiedgen_*` files remain a legacy
ten-stage adapter reference. Bedroom 4 has a truthful partial mapping at
`examples/bedroom4/site-profile.partial.yaml`; it leaves the full front-to-back
`layered_completion` executable instead of treating one pillow as a whole run.
Its old outputs are artifact-ready, but adoption remains blocked until the
missing original-video and per-stage input hashes are recovered.

## Quick start

```bash
uv sync --all-groups --extra geometry
npm install

uv run --extra geometry pytest
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
