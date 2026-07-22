# Round01 Front Pillow Recovery Route

This directory records the current recovery route for the corrected R1 front-pillow clean-plate blocker.

It is not an accepted clean plate and must not be used to start R2-R4. The route is generated from the verified `sam3_pillow_front` completion evidence plus the measured donor prefill report:

```bash
uv run video2world completion-route \
  examples/bedroom4/completion/layered-peel/round01_front_pillow/recovery/completion_evidence.json \
  --clean-plate-report examples/bedroom4/completion/layered-peel/cumulative-rgbd-reprojection/round01_front_pillow/output/multiview_prefill_report.json \
  --output examples/bedroom4/completion/layered-peel/round01_front_pillow/recovery/completion_backend_route.json
```

The expected recovery actions keep `allow_deeper_rounds=false` and require donor support, boundary QA, and constrained residual generation before any deeper layer can be promoted.
