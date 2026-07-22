# Round01 Front Pillow Recovery Route

This directory records the current recovery route for the corrected R1 front-pillow clean-plate blocker.

It is not an accepted clean plate and must not be used to start R2-R4. The route is generated from the verified `sam3_pillow_front` completion evidence plus the measured donor prefill report:

```bash
uv run video2world completion-route \
  examples/bedroom4/completion/layered-peel/round01_front_pillow/recovery/completion_evidence.json \
  --clean-plate-report examples/bedroom4/completion/layered-peel/cumulative-rgbd-reprojection/round01_front_pillow/output/multiview_prefill_report.json \
  --output examples/bedroom4/completion/layered-peel/round01_front_pillow/recovery/completion_backend_route.json
```

The expected recovery actions keep `allow_deeper_rounds=false` and require donor support, boundary QA, and constrained residual generation before any deeper layer can be promoted. The route can be converted into an auditable work order without generating or accepting any clean plate:

```bash
uv run video2world completion-recovery-work-order \
  examples/bedroom4/completion/layered-peel/round01_front_pillow/recovery/completion_backend_route.json \
  --clean-plate-report examples/bedroom4/completion/layered-peel/cumulative-rgbd-reprojection/round01_front_pillow/output/multiview_prefill_report.json \
  --output examples/bedroom4/completion/layered-peel/round01_front_pillow/recovery/recovery_work_order.json
```

The work order can then be converted into ordered recovery execution steps. This still does not run a model or generate clean plates:

```bash
uv run video2world completion-recovery-bundle \
  examples/bedroom4/completion/layered-peel/round01_front_pillow/recovery/recovery_work_order.json \
  --output examples/bedroom4/completion/layered-peel/round01_front_pillow/recovery/recovery_bundle.json
```

Before running recovery, preflight the bundle against local files:

```bash
uv run video2world completion-recovery-preflight \
  examples/bedroom4/completion/layered-peel/round01_front_pillow/recovery/recovery_bundle.json \
  --binding input_manifest=/Users/zhangyuxiang/Desktop/worksplace/video2world/examples/bedroom4/completion/layered-peel/cumulative-rgbd-reprojection/round01_front_pillow/manifest/cumulative_removal_manifest.json \
  --binding camera_info=/Users/zhangyuxiang/Desktop/worksplace/video2world/examples/bedroom4/assets-local/clean-scene-reconstruction-input-v23/camera_info.json \
  --binding source_rgb_frames=/Users/zhangyuxiang/Desktop/worksplace/Video2Mesh/tmp_remote_results/bedroom4_clean_plate_all_pillows_20260720/frames \
  --binding depth_arrays=/Users/zhangyuxiang/Desktop/worksplace/Video2Mesh/tmp_remote_results/holi_spatial_bedroom4_fresh_da3_sam3_pgsr_20260714_184217/scannetppv2/data/bedroom_4/depth_da3 \
  --binding physical_donor_exclusion_index=/Users/zhangyuxiang/Desktop/worksplace/video2world/examples/bedroom4/completion/layered-peel/cumulative-rgbd-reprojection/round01_front_pillow/manifest/donor_exclusion_index.json \
  --output examples/bedroom4/completion/layered-peel/round01_front_pillow/recovery/recovery_preflight.json
```

The current local preflight is `blocked_binding_semantics`. The RGB frames, camera info, cumulative manifest, and donor exclusion index cover `000048`-`000072`, but the Holi-Spatial `depth_da3` mirror only exposes `000000`-`000011`. Recovery execution must not start until `depth_arrays` is rebound to a depth directory or manifest that covers the same `000048`-`000072` frame IDs, or until an auditable camera/depth remapping artifact is materialized.
