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

