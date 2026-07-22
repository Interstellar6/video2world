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
