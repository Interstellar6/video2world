# Video2World Modeling

Independent, modular Video2World modeling. No existing `video2world` or `vnext` package is imported or invoked. Each module owns one artifact contract and runs its adapter in an isolated model interpreter. The same adapter works as a command provider or behind an independently registered HTTP worker.

## Task Boundary

All modeling inputs, intermediates, receipts and final manifests stay inside one task:

```text
outputs/<task_id>/
  inputs/                    copied source images and calibration
  stages/<module>/<run-id>/   immutable per-run model outputs and raw responses
  stages/<module>/           current invocation and stage receipts
  artifacts.json             role -> task-local artifact with evidence class and SHA-256
  task.json                  execution state and promotion blockers
```

The ABI is identical for all adapters:

```text
--task-dir <outputs/task_id>
--inputs <outputs/task_id/stages/module/inputs.json>
--outputs <outputs/task_id/stages/module/provider-artifacts.json>
```

An adapter publishes exactly its declared output roles only after producing and validating the files. Absolute external paths, symlink escapes, contract-only inputs to real models, mismatched hashes, and incorrect evidence/collision roles fail closed. A partial or failed stage is not a successful pipeline.

## Modules

| # | Module | Current Real Adapter | Main Outputs |
|---:|---|---|---|
| 1 | `scene_reconstruction` | calibrated COLMAP + camera-conditioned DA3 + fresh PGSR + TSDF | frames, cameras, depth, observed PLY/mesh |
| 2 | `scene_understanding` | local Qwen2.5-VL with optional object-crop class inventory | structured descriptions and frame-local group boxes |
| 3 | `component_segmentation` | full native SAM3-I, all-instance class queries | `component_mask_candidates`, `physical_instance_hypotheses` |
| 4 | `object_lifting` | calibrated depth and multi-view geometric association | resolved `component_masks`/`physical_instance_tracks`, object PLY, carved PGSR, rejection report |
| 5 | `clean_plate` | Responses controller + `image_generation` edit tool | generated clean-plate keyframes and request journal |
| 6 | `background_reconstruction` | fresh DA3/PGSR/TSDF on fixed-camera clean plates | generated background candidates |
| 7 | `component_assembly` | same-camera verified parent RGBA extraction | observed object views, component annotations and completion plan |
| 8 | `orbit_video` | Gaussian orbit rendering + official FixAnything | three 360-degree elevation videos |
| 9 | `geometry_completion` | Stream3D + SAM3D, then observed-anchor registration | generated meshes with validated Sim(3) placement proof |
| 10 | `mesh_postprocess` | mesh repair/simplification, UV texture baking and CoACD utilities | repaired visual mesh and convex hulls |
| 11 | `physics_estimation` | Qwen-VL with explicit uncertainty | physical OBJ and property sidecars |
| 12 | `scene_recomposition` | shared per-object Sim(3), separate visual/collision roles | auditable scene bundle |

Segmentation does not establish cross-view physical identity or edit the parent mask. Lifting alone resolves candidates: it independently validates per-frame semantic group unions and individual component tracks against calibrated depth, using the same existing geometry thresholds. A group may extend a parent only with accepted group geometry and an explicit category/group membership policy. Passing group geometry does not certify individual identities. Only independently accepted individual tracks may export component PLY, with at least two positive views and no visible-negative votes. Final whole-parent association and exclusive Gaussian ownership remain mandatory.

Assembly consumes the lifting-bound parent mask in the same source camera. Components are annotations only; they never alter alpha or paste pixels from another viewpoint. Assembly creates a fresh run directory, validates mask hashes, and preserves earlier successful artifacts. Missing and hidden surfaces remain incomplete until the completion provider runs.

Orbit conditioning verifies the assembly/lifting/PLY bindings and records every camera and rendered-frame hash. One common focal adjustment across all 183 conditioning frames improves framing without changing Gaussian geometry or camera poses. Images are rerendered at the final calibrated resolution, not enlarged from thumbnails. Original conservative renders remain available. Geometry completion validates three distinct closed conditioning trajectories, then estimates fresh generated-image cameras/depth and checks camera consistency before Stream3D; conditioning camera poses are never declared measured generated-pixel geometry.

A geometrically consistent component-mask track does not prove that its mask contains exactly one physical instance. Native SAM can merge adjacent pillows, and a class label can still be wrong after geometric association. Keep physical cardinality and semantic correctness unverified unless separately checked; do not infer a pillow count from the number of exported tracks.

The deployed mesh postprocessor is a utility toolchain, **not a verified execution of the learned 3D-Fixer model**. The official [3D-Fixer](https://github.com/HorizonRobotics/3D-Fixer) takes observed image/mask context and partial geometry; it belongs to a completion-provider boundary, not the repair/simplify/UV/CoACD utilities. The independent `three_d_fixer.py` adapter is wired as the opt-in `observed_context_completion` provider, selected per process by `scripts/three_d_fixer_cli.py`; its outputs stay `generated` and `promotion_allowed=false` because no visual acceptance gate exists. TRELLIS/TRELLIS2/Hunyuan3D and VGGT-Omega remain alternative boundaries, not interchangeable working bindings.

### Robustness Policies

Three defects that used to abort a whole stage are now handled where they occur and reported instead of hidden, because each one is a property of the provider rather than of the scene:

* **Non-finite scene Gaussians.** PGSR leaves a small minority of splats non-finite when a few view-inconsistent points never converge (2.7% of the 292k-splat ScanNet++ DSLR cloud). A NaN position cannot be rasterised, so `world_modeling.gaussian_io.read_gaussian_rows` drops those rows and every reader records the count (`source_gaussian_rows_dropped_non_finite`). A table with nothing finite left is still fatal.
* **Bound pymeshfix.** Its component joining is superlinear: a 100k-face completion mesh ran for two hours without converging, and its result is usually rejected by `meshfix_verdict` anyway. Above `PYMESHFIX_FACE_LIMIT` faces the repair is refused and recorded as `skipped_mesh_over_<limit>_faces:<n>`; watertightness is preferred, not required, and CoACD supplies collision geometry independently.
* **Boxes that overshoot the raster by a few percent.** A model that returns a boundary-touching box a few percent wider than the frame is clamped and the original coordinates are kept in `bbox_clamped_from`; anything outside by more than `BOX_CLAMP_TOLERANCE_FRACTION` of the side stays fatal. Object labels outside the declared `[a-z][a-z0-9_]{0,63}` grammar are rewritten with the original preserved in `source_vlm_object_id`, a repeated instance inside one frame is dropped into `duplicate_objects_dropped`, and two distinct instances that share a label are suffixed and recorded in `identifier_rewrites`. The crop inventory pass keeps the strict grammar, because its caller can afford a retry.

### Marginal Quality Measurements Are Recorded, Not Fatal

A measurement that lands a fraction of a percent from a target is evidence about a provider, not proof that an object is absent, and three such gates used to reject an entire ScanNet++ DSLR run whose geometry was otherwise sound. They were changed **by explicit operator decision** to report instead of block, each with a coarse floor that still rejects a genuinely wrong result. None of them raises `promotion_allowed`, and every one of them writes the measurement, the target and the floor into the receipt:

| Measurement | Target | Floor that still rejects | Where it is recorded |
|---|---|---|---|
| Cross-view visibility conflict inside one lifted track | none allowed | identity still requires every observation to be reachable through accepted pairs | `tracks[].component_association.visibility_conflicts`, `association_report.visibility_conflicts` with `policy: recorded_not_blocking` |
| Selected-view DA3 vs isolated-object support in the 3D-Fixer | 0.9 | `SUPPORT_FLOOR = 0.5` | `selected_view_support` (`below_target`, `support_fraction`) |
| Heldout original-camera reprojection p95 | `max_p95_pixels` (8) | `REPROJECTION_REJECTION_FACTOR × target` (32 px) and `source_inside_observed_mask_fraction ≥ 0.9` | `reprojection` report (`above_target_frames`, `worst_p95_pixels`) |

Measured on `00a231a370`: the bed track had 1 conflicting pair out of 66 with a fully connected graph; the 3D-Fixer selected view measured 0.8971 support; placement measured a 13.07 px p95 with 0.998 of its pixels inside the observed mask. The completion stage also records per-object rejections in `rejected_objects` and completes the objects that are usable, matching how `object_lifting` reports rejected tracks; a run where nothing survives is still fatal.

Generated clean plates and their reconstruction remain `generated`, with collision eligibility and promotion disabled. Clean plates remove only lifting-accepted objects, protect rejected-object masks, retain raw generated images, and copy every final unmasked pixel exactly from the calibrated source. Unsegmented original views are excluded so background reconstruction cannot train removed objects back into the scene.

The default scene visualization is carved PGSR plus object PLY. OBJ/CoACD are separate physical proxies using the same verified object transform. Unknown alignment, metric scale, physical properties or background collision are reported rather than guessed. A completed process is not visual or simulation acceptance.

## Run Real Adapters

`profiles/seetacloud.json` binds all 12 real adapters to explicit paths on the configured SeetaCloud machine. **It is the default profile, not a contract smoke test.** Its current stage-1 input is an undistorted calibrated dataset containing `images/` and `sparse/0/`, not an uncalibrated video.

### One Command From a Dataset to a Verified Export

`scripts/run_dataset.py` needs only a calibrated dataset directory and an output directory. The profile, the learned completion provider and the skipped background branch are all defaults, and `--export-dir` appends the delivery step and its independent verification:

```bash
python3 scripts/run_dataset.py \
  --source /path/to/dataset --output-dir /path/to/runs --export-dir /path/to/delivery
```

`--output-dir` is either the run root, in which case the task lands in `<output-dir>/outputs/<task_id>`, or the task directory itself when it already ends in `outputs/<task_id>`; the resolved task path is printed before anything runs. The task id defaults to the dataset directory name and an unusable name is refused rather than silently defaulting, because two datasets must never share one task directory. Verified on `00a231a370`: one invocation of `run_dataset.py` with only `--source` and `--output-dir` completed all nine modules (`PIPELINE_EXIT=0`) into `/root/autodl-tmp/oneclick-run`, and the same command with `--export-dir` wrote and independently verified the delivery layout (`EXPORT_VERIFY_EXIT=0`, `valid: true`). Re-running it against an existing task resumes instead of starting over.

The one input this launcher cannot invent is the dataset itself: stage 1 needs `images/` plus `sparse/0/`. A ScanNet++ DSLR release must be converted first with `build_dslr_source.py` above, and any other capture must bring its own undistorted COLMAP reconstruction.

```bash
cd /root/autodl-tmp/video2world-modeling
/root/miniconda3/bin/python scripts/run_pipeline.py my-scan \
  --source /absolute/path/to/calibrated_dataset --through object_lifting
PYTHONPATH=src /root/miniconda3/bin/python -m world_modeling validate my-scan
```

### Prepare a Source Dataset From a ScanNet++ DSLR Capture

Stage 1 consumes `images/` plus `sparse/0/`, so a raw ScanNet++ DSLR release must be converted first. `scripts/build_dslr_source.py` takes the **poses** from `colmap/images.txt` (already COLMAP world-to-camera, sharing the world frame of `points3D.txt`) and the **undistorted PINHOLE intrinsics** from `nerfstudio/transforms_undistorted.json`, because the shipped `resized_undistorted_images` are undistorted and the `OPENCV_FISHEYE` `colmap/cameras.txt` therefore does not apply to them. The two files use different axis conventions, so the script compares their camera-centre distance matrices instead of their poses; nothing is written unless they describe one capture and the sparse points project inside the frames through those poses and intrinsics.

A DSLR release is a walk through the whole capture, while stages 1 and 4 expect one coherent sequence: DA3 multi-view depth and PGSR need tight baselines, and the lifting stage's all-pairs identity test needs views that actually share surface. `--window-strategy coverage` therefore scores contiguous segments by how many triangulated points at least three of their frames observe together, and takes the best one instead of spreading the selection over the whole walk.

```bash
python3 scripts/build_dslr_source.py \
  --dslr-dir /path/to/scannetpp/<scene_id>/dslr \
  --output /tmp/<scene_id>-source --window-strategy coverage --window-size 8 --window-stride 1
```

Measured on `00a231a370`: spreading 60 frames over all 779 gave sparse-point visibility 0.544 (median) and a 40-observation bed track whose pairs disagreed 223 times; the best contiguous 12-frame segment raised visibility to 0.819 and left a single conflicting pair; the best contiguous 8-frame segment lifted every one of its tracks. Prefer the shortest segment that still carries three or more views of the target.

### Skipping the Background Branch

`--skip-bg-recon` drops `clean_plate`, `background_reconstruction` and `scene_recomposition` from whatever the profile declares, so the rest of the recipe runs unchanged against the same source. It exists on both entry points and records nothing else: the task reports the nine modules it actually ran, with no retired stages.

```bash
/root/miniconda3/bin/python scripts/three_d_fixer_cli.py run my-scan \
  --profile profiles/three-d-fixer.skipbg.json --skip-bg-recon
```

`profiles/three-d-fixer.skipbg.json` is the twelve-stage recipe with the learned 3D-Fixer bound where the recipe names `geometry_completion`; the provider version of every other stage is byte-identical to `profiles/seetacloud.json`. A profile used through `scripts/three_d_fixer_cli.py` must bind `observed_context_completion` and must not bind `geometry_completion`: the registry swap rejects either mismatch, and `world_modeling validate` (without the fixer CLI) will keep reporting the swapped module as unregistered, so validate such tasks with `scripts/three_d_fixer_cli.py validate` instead.

For an explicit interface-only test without any model inference:

```bash
python3 scripts/run_pipeline.py contract-demo --source /absolute/path/to/source \
  --profile profiles/contract.json
```

The contract profile is never a source of real geometry. Real outputs stay `promotion_allowed=false` until independent appearance, geometry, novel-view, support/contact and collision gates pass.

The original task recipe is preserved at `outputs/bedroom4-fresh-modular-20260907/run-profile.json`, with all 12 bindings and the original stage-1/2 arguments. Stage 3 was explicitly switched to native SAM3-I. `run-profile.initial.json` preserves the initial profile byte-for-byte. The current bed run uses `run-profile.pillow-group-policy.json`, which only adds the explicit `bed/pillow_group=bedding` membership policy to that recipe. The task recipe retains its validated DA3 phase-resume path; the generic profile deliberately has no hardcoded task resume.

The early command receipts predate source-version recording. Their input/output hashes remain verifiable, but their historical source identity was not recorded and is never backfilled. Continue this existing task with `run-stage` for the desired later module, consuming its verified upstream artifacts. A full `run` from stage 1 will not silently reuse legacy command receipts as proof of the current code and can rerun expensive stages.

## Registered Module Services

Workers expose discovery and asynchronous jobs over the shared task store. They do not load GPU models at startup; model argv and environments are selected only when a job is submitted. Start and register every configured module, or select individual modules with repeated `--module` flags:

```bash
python3 scripts/serve_modules.py start modeling-services --dry-run
python3 scripts/serve_modules.py start modeling-services
python3 scripts/serve_modules.py status modeling-services
```

The launcher defaults to loopback, skips occupied ports without touching their owners, refuses to replace existing deployments, and stores its profile snapshot, logs, process identities, registry and ready-to-use `services.pipeline.json` inside `outputs/modeling-services/`. That generated profile uses the same service-provider schema as `profiles/services.pipeline.json`, with the correct deployment-local registry path.

```bash
PYTHONPATH=src python3 -m world_modeling discover \
  --registry outputs/modeling-services/services.registry.json
PYTHONPATH=src python3 -m world_modeling run my-scan \
  --profile outputs/modeling-services/services.pipeline.json
python3 scripts/serve_modules.py stop modeling-services
```

Stop verifies host/PID/start-time ownership, sends SIGINT only to its own worker PIDs, and lets active model jobs drain. It does not kill process groups, unrelated jobs, or force-kill a slow provider. `draining_or_identity_unknown` requires inspection; a new deployment ID never silently terminates the old one.

For manual control, the same API is available through `serve MODULE --profile ... --port ...`, `register --registry ... --endpoint ...`, `discover --registry ...`, and `run-stage TASK MODULE --profile ...`. `profiles/services.pipeline.json` is a manual template whose registry path is relative to that profile.

Module registration is independent of the fixed built-in list: install a package exposing the `world_modeling.modules` entry-point group with `ModuleSpec` values or a callable returning them, then add its command binding. `profile.pipeline` chooses the active modules; dependency ordering and missing producers are checked before execution. Adding a backend for an existing module only changes its provider binding. Changing live source files or provider recipes requires worker restart and rediscovery.

Both command and service providers record source revisions. Static local imports, sibling helper scripts and explicit `revision_files` participate in the cache key; an unrelated adapter change does not invalidate a command's reconstruction cache. Dynamic entrypoints require explicit revision files. Declare dynamically loaded backend source, model metadata and environment lock files there when those versions are part of a provider's identity. Source resolution fails before replacing an existing successful receipt, and code changes during execution prevent publication.

### Opt-In Learned 3D-Fixer

This alternative consumes original observed RGB, masks, cameras and depth, not FixAnything videos. Its explicit bootstrap replaces `geometry_completion` with `observed_context_completion` only in that process's registry, then reuses the same CLI, HTTP discovery, registration and invocation implementation. The normal 12-module CLI is unchanged. Both worker and client must use the opt-in entry point:

```bash
python3 scripts/three_d_fixer_cli.py discover
python3 scripts/three_d_fixer_cli.py serve observed_context_completion \
  --profile profiles/three-d-fixer.example.json --port 8138
python3 scripts/three_d_fixer_cli.py run-stage my-scan observed_context_completion \
  --profile profiles/three-d-fixer.example.json
```

The example is stage-only and requires actual deployment paths. Seven dedicated checkpoints total 5,345,250,948 bytes; reusable TRELLIS weights cannot replace them — the TRELLIS base copies have the same sizes but different bytes, so the official SHA-256 preflight rejects them. Preparation-only runs never publish completion roles. The actual model adapter preserves native camera-to-world/normalized-intrinsics semantics, validates observed source/mask hashes and requires heldout registration before publication.

`profiles/three-d-fixer.seetacloud.json` is the deployed SeetaCloud binding. Three deployment facts are not optional: the provider needs the MoGe **v2** checkpoint (`Ruicheng/moge-2-vitl`); the MoGe v1 checkpoint loads but its config lacks the `neck` argument `MoGeModel` requires. The vendored FlexiCubes submodule imports `kaolin.utils.testing.check_tensor` only for debug shape assertions, so `external/three-d-fixer-shims` supplies that one pure helper instead of building kaolin. `diff_gaussian_rasterization` must be built into the provider interpreter; without it the model still samples correctly but GLB texture baking fails.

The raw model surface drifts from the multi-view observation even where the object was observed (heldout reprojection p95 median 10.77 px, 43 of 49 cameras over the 8 px bound), because the model is conditioned on one observed view. `--observed-refinement` adds an explicit, off-by-default data-fit step that moves mesh vertices only toward measured points inside a radius (default `2.5 * registration threshold`) with Laplacian smoothing; unobserved regions move only through Laplacian coupling and no geometry is invented. With it the same stage passes its own gate at p95 median 4.55 px with 0 of 49 cameras over the bound, and the step is recorded in `observed_region_refinement` on every published object. It is a fitting step, not a relaxation of any gate: the registration and reprojection thresholds are unchanged.

## Export Assets

`scripts/export_assets.py` maps a task's hash-verified artifacts into the video2world asset layout (`datasets/<scene>/`, `scene/point_cloud_3dgs.ply`, `scene/point_cloud_simple.ply`, `scene/mesh.ply`, `object/background.glb` for the observed scene mesh, `object/background_generated.glb` for the generated background candidate, `object/object_version_1|2/<object_id>/`, `recomposition/`):

```bash
PYTHONPATH=src python3 scripts/export_assets.py my-scan \
  --export-root /path/to/video2world-export-assets/bedroom_4 --scene-name bedroom_4_47_56
```

Every copied artifact is re-verified against its task receipt first, and `export_manifest.json` records the source path, SHA-256 and evidence class, so generated candidates are never presented as observed geometry. The exporter reports `partial` while `object_version_2` has no completed object; scene-only output is not a finished export.

`scripts/verify_export.py` independently re-checks a delivered directory against its own manifest: it re-hashes every recorded file, loads every mesh, and re-asserts positive volume, finite vertices, and convex watertight collision hulls. It reports failures instead of repairing them and states that structural integrity is not visual or simulation acceptance.

```bash
python3 scripts/verify_export.py --export-root /path/to/video2world-export-assets/bedroom_4 [--skip-geometry]
```

## Delivery Layout

The delivery is the only thing a consumer receives, so every documented capability has to be a file in it. `export_assets.py` writes this shape:

```text
<delivery>/
  datasets/<scene>/                 source images and calibration
  scene/point_cloud_3dgs.ply        scene Gaussians (PGSR); non-finite rows are dropped and counted
  scene/point_cloud_simple.ply      scene surface points
  scene/mesh.ply                    scene surface, decimated to the face budget, vertex-coloured
  scene/depth/                      per-frame 16-bit PNG depth plus manifest.json
  object/background.glb             scene surface with a baked texture, or vertex colours as fallback
  object/background_texture.png     the baked atlas
  object/parts/<object>/            observed per-component parts, one PLY each, plus manifest.json
  object/object_version_1/<id>/     completed mesh and textures plus asset_report.json
  object/object_version_2/<id>/     EmbodiedGen v2 asset: GLB, textured OBJ/MTL/PNG, collision/ hulls
  qa/                               previews, comparison and QA records
  web-demo/                         three.js viewer over the delivered assets
  export_manifest.json              every delivered file with source path, SHA-256 and evidence class
```

`asset_report.json` sits in each object directory and lists the stages that produced that object — role, provider, evidence class and source frame ids — so an object can be traced without opening the task registry. Vertex colours are written as a normalised unsigned-char array on purpose: PLY and GLB writers silently drop them otherwise, and `verify_export` re-reads the files rather than trusting the writer.

Three scene assets have a stated budget because an unbounded one is not a delivery: the scene surface is decimated to `--scene-face-budget` faces, the background to `--background-face-budget` faces with a `--background-texture-size` atlas baked from the calibrated frames, and the depth layer writes 16-bit PNGs per frame with a manifest. Decimation uses `vtkDecimatePro`; `vtkQuadricDecimation` refuses to reduce an open TSDF surface because it preserves boundary edges (measured: 12.79M to 12.19M faces). The scene surface is loaded through VTK, not trimesh, for the same practical reason: 2.2s against minutes on a 12.8M-face surface.

### The Provider Mapping Behind It

| Level | Recipe | Bound provider |
|---|---|---|
| Scene | Holi-Spatial: DA3/VGGT-Omega depth, PGSR Gaussians at `--pgsr-iterations 30000`, TSDF surface | `scripts/scene_reconstruction.py` |
| Detection | Qwen2.5-VL frame inventory with `--component-inventory-policy tolerant` and `--object-hints` | `scripts/qwen_understanding.py` |
| Segmentation | native SAM3-I, all-instance class queries | `scripts/component_segmentation.py` |
| Video | FixAnything orbit video, three elevations per object | `scripts/orbit_video.py` |
| Completion | Stream3D + SAM3D over the generated views | `scripts/geometry_completion.py` |
| Objects | EmbodiedGen v2: splat renders, `MeshFixer`, `TextureBaker`, `save_mesh_with_mtl`, CoACD | `scripts/embodiedgen_assets.py` |

`profiles/embodiedgen-stream3d.skipbg.json` binds exactly this and is the default of `run_dataset.py`. It replaces the builtin `mesh_postprocess` utilities with the EmbodiedGen v2 toolchain, which is what turns "a mesh and a splat" into a textured OBJ/MTL/PNG triple plus convex collision proxies; the builtin provider stays bound in `profiles/three-d-fixer.skipbg.json`.

Scene density is not an iteration-count decision. Measured on bedroom_4 at 50 frames and resolution 2, `--pgsr-iterations 6000` delivered 453,381 Gaussians and `--pgsr-iterations 30000` delivered 437,419: densification stops at `--densify-until-iter` and opacity pruning continues after it, so more iterations train a better fit of a scene of the same size. Density is a separate control — a lower `--densify-grad-threshold` (default 0.0002) and a longer densify window — which is why the provider exposes both rather than only the iteration count. The reference delivery's scene model held 1,331,069 Gaussians.

### Cameras for the Generated Views

Stream3D needs a pose and a calibration per input view, and the stage first tries to measure them: DA3 is run on the generated frames alone (`extrinsics=None`), and its trajectory is aligned to the conditioning orbit with a Sim(3) fit whose error gates are the registration gates. That measurement is rejected on a turntable orbit, and the reason is structural rather than a threshold: DA3 explains a rotating object as a moving camera, so on the six degree-of-freedom-free orbit renders it estimates almost no baseline. Measured on the bedroom_4 bed, the estimated camera-centre span was 1.9 units against 4.5 in the conditioning trajectory, with each orbit's four views collapsing into a 0.2-unit cluster, and the Sim(3) fit then found fewer than three supported correspondences.

`--generated-camera-fallback conditioning` therefore lets the stage keep the depth the generated view actually predicts while taking the pose and calibration of the conditioning frame that view was rendered from — generated frame `i` is the conditioned edit of conditioning frame `i`, so those are the exact cameras of the view, not an approximation of them. The switch is never silent: `da3_world_to_camera` and `da3_intrinsics` keep the rejected estimate on every frame, `generated_camera_fallback` records its failure reason and both measured spans, `camera_estimation` becomes `conditioning_trajectory_of_registered_generated_frames`, `conditioning_poses_used` turns true, and the dataset declares the conditioning scene frame and units so the reconstructed object lands in the scene frame rather than in a hallucinated one. The default remains `fail`, which is what a profile gets when it does not ask for the fallback, and a dataset that already declares the conditioning trajectory is re-verified against it exactly rather than re-estimated.

## Comparing Deliveries

`scripts/compare_deliveries.py` measures two deliveries item by item — total size, every scene layer with its vertex/face counts and whether it carries colour, objects per version with mesh, splat and hull counts, delivered parts, advertised previews and whether the files exist, and the viewer — and checks the candidate against a delivery contract: a scene surface, Gaussians, depth maps, completed objects, a mesh and a splat per object, collision hulls per object, a textured background, observed parts, at least four preview kinds, and a viewer.

```bash
python3 scripts/compare_deliveries.py --reference /path/to/old --candidate /path/to/new \
  --report /path/to/new/qa/comparison.json --require-contract
```

The contract is deliberately stricter than the reference it was written against: that export carried no collision hulls for any of its 15 objects and no depth maps, so it fails the contract the current delivery satisfies. A violation is reported per line and `--require-contract` turns it into an exit code instead of a claim.

A delivery can also be described where it lives and compared where the reference lives, which avoids moving hundreds of megabytes between hosts:

```bash
python3 scripts/compare_deliveries.py --describe <delivery> > candidate.json   # on the GPU host
python3 scripts/compare_deliveries.py --reference <old> --candidate candidate.json --require-contract
```

`run_dataset.py --export-dir <path> --compare-with <old>` does the export, the independent verification and this comparison in one invocation.

## Credentials and Readiness

Credentials are inherited through environment variables, never written into profiles or launcher receipts. `--token-env` on the launcher is optional service authentication; it is distinct from clean plate's `PLBBL_API_KEY`. Load provider secrets in the launching environment without printing them.

`clean_plate` also has an offline backend: `--local-lama-model <inpainting_lama_2025jan.onnx>` runs the official OpenCV LaMa ONNX asset through OpenCV DNN behind the same Responses-shaped transport, so the journal, budget, compositing and QA logic are unchanged while no credential or network is used. `profiles/seetacloud.local-clean-plate.json` binds it. The endpoint and token become unused in that mode, and the outputs stay `generated` candidates exactly as with the API. Accepted-mask boundary slivers shared with an unaccepted object are resolved by assigning the disputed pixels to the protected object, bounded by `--max-ownership-overlap-fraction` (default 0.01) and recorded as `protected_ownership_pixels`; a larger overlap still fails.

The clean-plate profile defaults to a cumulative API budget of **0**. After authentication and first-frame approval, explicitly raise `--max-api-requests` in a task/deployment recipe. A budget of 1 can produce one candidate and then stops without publishing an incomplete 22-frame batch. Submitted/uncertain requests are not silently retried. An intentional retry requires its exact journal `--retry-request-id` and sufficient additional budget.

### Current Bedroom 4 Evidence

The current 50-frame Qwen -> native SAM3-I -> lifting -> assembly chain is recorded at `qa/full50-new-boundary-20260907T144208Z`. It reuses this task's hash-verified calibrated reconstruction, not an earlier object's geometry. Qwen identified bed/nightstand, but missed lamps. Native SAM3-I strictly loaded all 3,403 checkpoint keys and produced 641 accepted candidate masks with nine rejected proposals.

The explicit pillow-group policy plus independently accepted group geometry allowed the whole bed to pass all 1,225 frame-pair checks with zero conflicts. The observed bed contains 147,448 Gaussians; the carved scene retains 308,227, exactly partitioning the original 455,675. Assembly produced eight observed views. No geometry threshold was relaxed, and the original profile and earlier failed runs remain intact. Component-track counts do not establish pillow cardinality.

Nightstand remains rejected with 286 conflicts: the same Qwen ID mixed a right-side tabletop edge, a right-side lamp and the left nightstand. Even the left-side subgroup retains conflicts. A separate two-frame lamp recall probe found two actual lamps, but it did not publish masks, physical tracks or task roles. The earlier lamp/nightstand and six-view bed runs are historical QA only; they are not the current full-50 result.

FixAnything is restored on the server: all 15 base files (47,067,183,559 bytes) matched their acquisition hashes, and all 400 LoRA target shapes matched. It requires the dedicated `fixanything-bedroom4-poc-20260829/env/bin/python` environment. The actual current bed stage `stages/orbit_video/run-fu6dy26_` completed all three elevations (10/25/40 degrees), each with 61 distinct decoded frames at 832x480 and 15 fps. It used 10 inference steps and no trusted clean anchors. Original and framed conditioning, cameras, logs and videos are preserved.

All three current bed videos are **rejected as reconstruction inputs**: invented supports, detached artifacts and inconsistent bed structures/textures across elevations prevent treating them as one object's multiview evidence. `qa/fixanything-full50-three-orbits` contains the bound visual review and interface validation. Generation/decode success is not appearance or 3D acceptance; Stream3D was not run on these clips. The earlier attractive bed candidate used a different, more complete standalone asset, black-background conditioning and clean frames 0/60. That comparison does not establish which individual setting caused the difference.

The pipeline now runs end to end for the bed. PLBBL authentication still returns HTTP 401 and no paid image request was sent; instead `clean_plate` runs the offline LaMa ONNX transport, producing 50 generated clean plates, after which `background_reconstruction` produces a fresh DA3/PGSR/TSDF background. The opt-in learned 3D-Fixer publishes `completion_candidates` and `completed_object_meshes` (status `generated_and_geometrically_registered`, `generated_orbit_used=false`) with an accepted registration and its own heldout reprojection gate at p95 median 4.55 px and 0 of 49 cameras over the bound; the raw surface had failed that gate systemically (p95 median 10.77 px, 43 of 49 over) because the model is conditioned on one observed view, and `--observed-refinement` is a data-fit step rather than a threshold change. 10,433 saturated opacity logits were clamped finite. `mesh_postprocess` executes with a guarded pymeshfix step: pymeshfix had collapsed the completion mesh into a thin sheet (volume 0.00102 against 0.01169, Y extent 0.108 against 0.616), so its result is now accepted only when it closes the surface without moving bounds more than 20% of the diagonal or changing volume by more than 2x; otherwise the pre-repair mesh is kept and `meshfix_outcome` records the rejection. The repaired bed keeps 100,000 faces at volume 0.01544 and full bounds. `physics_estimation` needed a prompt reorder (ending the prompt with the description JSON made Qwen2.5-VL-3B continue that JSON instead of answering). `scene_recomposition` then reports `assembled_candidate` with zero errors, using the carved observed PGSR as the visual background. Stream3D completion remains fail-closed: `geometry_completion` rejects the orbit clips before Stream3D with `generated camera alignment failed ... RANSAC found fewer than three supported correspondences`. A least-squares Sim3 between the DA3-recovered generated cameras and the conditioning trajectory shows those clips are off by 19.6% of the orbit radius (residual p95 13.5 units against a 0.49 unit threshold), so this is a generation-quality failure, not a threshold artefact. A clean-conditioning experiment (`--conditioning-radius-clip 0.25 --conditioning-dc-only`, available behind explicit flags and off by default) made it **worse**: the trajectory error rose to 45.3% and the fit needed a reflection, so the deployed conditioning remains the original. Every published role keeps its evidence class: completion, clean plates and background stay `generated`, and `promotion_allowed` remains false because no visual, geometry, collision or source-camera acceptance gate exists in the orchestration contract. A completed process is still not visual or simulation acceptance.

### Scene-Level Delivery Preparation

The reconstruction hands back a raw TSDF surface: on bedroom_4 that is 12,798,127 faces and 260 MB, which no viewer opens. Export therefore prepares the scene level before writing it, and records what it did in `export_manifest.json`:

* `scene/mesh.ply` and `object/background.glb` are decimated to `--scene-face-budget` faces (default 1,000,000). `vtkDecimatePro` is used because `vtkQuadricDecimation` refuses to reduce this open surface at all -- it preserves boundary edges, and a TSDF surface is mostly boundary. Measured: 12.8M faces to 1.0M in about 130 s, keeping the reconstructed vertex colours.
* Readers go through VTK rather than trimesh, which has to walk 12.8M faces in Python; `vtkGLTFReader` reads the same file in 2 s.
* Vertex colours are normalised to an unsigned-char RGB active scalar array before writing. Both writers silently drop colours of any other shape, which is invisible until someone opens the delivered file.
* `scene/depth/` gains one 16-bit PNG per frame plus `manifest.json`, carrying the linear scale, offset, units and the hash of the raw float array each image came from.

Export now needs `vtk`, `trimesh` and `PIL`, so run it with an environment that has them (the EmbodiedGen environment does):

```bash
/root/autodl-tmp/envs/embodiedgen-011d815ea393-py310-cu124/bin/python scripts/export_assets.py \
  my-scan --export-root /path/to/delivery --scene-name my_scene
```

### Observed Object Parts

Lifting already carves every geometrically verified component track of an object, so the parts a delivery wants -- bedding, a headboard, pillows, a lamp shade -- exist as observed geometry before any generative completion touches them. Export copies each one to `object/parts/<object_id>/<object_id>_part_NN.ply` (ordered by component id) and writes `object/parts/manifest.json` with the component id, the lifting association status, both hashes and the evidence class `observed`. A component PLY whose hash no longer matches its lifting receipt is refused rather than shipped.

Measured on bedroom_4: seven verified parts (bed 4, lamp 2, plant 1). They are observed geometry, not generated assets: completing and texturing them per part is a separate step that has not run.

### Delivery Viewer

Export also writes a `web-demo/` project next to the assets: a three.js page that
loads the scene layers, the background and each object's mesh and Gaussian splat,
with the object list generated from the export manifest (`--skip-viewer` opts
out). Nothing about the scene is hardcoded, so the viewer always matches what was
actually delivered.

```bash
cd web-demo && npm install && npm run build      # writes web-demo/dist
python3 -m http.server 8000                      # from the export root
# open http://127.0.0.1:8000/web-demo/dist/index.html
```

Asset URLs are absolute from the server root, and `?root=<prefix>` overrides that
if the viewer is hosted somewhere else. Meshes load as glTF; splats load as
coloured points with the DC term of the spherical harmonics decoded in the
browser, so this is a QA viewer rather than a photoreal splat rasterizer.
Verified against a real export: the built page serves, `assets.json` resolves to
the delivered layers and objects, and the referenced mesh/background URLs return
200 from a static server on the export root.

### Delivery Previews

Export also renders the QA images a delivery needs to be judged without opening a 30 MB mesh (`--skip-previews` opts out). `scripts/render_previews.py` draws them with a CPU point painter rather than OpenGL, because a headless box has no display: point clouds are perspective-projected and painted far-to-near, and detections come from the observed frames through PIL.

| Image | What it shows |
|---|---|
| `scene/scene_overview.jpg` | the delivered surface from three sides (top, front, back) |
| `object/object_layout_preview.png` | top-down view of the observed scene Gaussians |
| `object/object_detect_preview.png` | detected boxes and labels on their own source frame |
| `object/object_cutout_preview.png` | the detected crops side by side |
| `object_version_2/<id>/gaussian_turntable_overview.jpg` | six-view turntable of the object's own splat |
| `object_version_2/<id>/glb_mesh_overview.jpg` | the same turntable for the delivered mesh |

`qa/previews.json` records each image's hash, its painted fraction and every preview that was skipped with the reason. A preview that renders nothing is refused instead of shipped: an empty image looks like evidence.

### Scene Recipe

The reference delivery trained PGSR to iteration 30000 with a 7000 fallback, and its scene Gaussian cloud carries 871,317 points. The deployed profiles therefore ask for `--pgsr-iterations 30000` with `--max-frames 0`; `scripts/scene_reconstruction.py` scales the densification window with that number (`densify_until_iter = min(15000, iterations // 2)`) rather than hardcoding it, so a longer schedule actually densifies instead of only refining. `tests/test_profiles.py` pins this so a profile cannot quietly fall back to a short schedule.

Scene delivery then decimates and textures that reconstruction: `--scene-face-budget` (default 1,000,000) for the delivered surface and a separate `--background-face-budget` (default 100,000) with a baked 2048 texture for `object/background.glb`.

## Verify

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 scripts/serve_modules.py start dry-check --dry-run
```

CPU/HTTP synthetic fixtures test contracts and process management without invoking model inference. Model-specific tests report skipped numerical dependencies separately from passing assertions.

The current complete suite contains 360 tests. The remote base interpreter ran all 360 with eight dependency skips and zero failures after the 2026-09-10 source sync. A macOS `tar` sync had copied 53 AppleDouble `._*` sidecars into the repository; `source_revision` then refused to parse them (`source code string cannot contain null bytes`) and 24 tests errored. The sidecars were deleted and the suite returned to green, so always sync source with `--exclude='._*'`. Receipts and exact skip reasons for the earlier 349-test run are at `qa/final-verification-20260908` in the Bedroom 4 task. These checks do not establish model inference or visual acceptance.
