# Provenance-aware layered clean-plate compositor

Implementation: `scripts/compose_layered_clean_plate.py`.

This is an interface and tested synthetic fixture only. No Bedroom4 final clean
plate is claimed here.

## Input contract

The input manifest kind is
`video2world.layered_clean_plate_compositor_input`. Each frame supplies:

- an explicit `round_kind`: `object_peel` or `final_background`;
- source RGB and cumulative removal mask;
- the one newly removed object plus the ordered downstream objects that must remain;
- for `object_peel`, calibrated donor-prefill RGB, exact measured mask, and fused
  measured depth;
- zero or more depth-ordered downstream PBR RGBA/depth renders;
- one structural-background RGBA/depth render;
- hash receipts for the measured donor report and every render.

The measured mask must equal `removal_mask AND NOT donor_report.residual_mask`.
The measured depth must be stored in the donor report with role
`fused_multiview_measured_depth`. The report and receipt must state that
generated pixels and ProPainter pixels are both zero in the measured class.

Round 1 requires `source_contract.role=original_observed_rgb`. Round N greater
than 1 requires `source_contract.role=previous_layered_composite`, the previous
report and receipt hashes, and source RGB hashes equal to the previous round's
composite RGB outputs. The previous round must have a complete provenance
partition with zero unresolved pixels. `newly_removed_object_id` must equal the
last item in cumulative `removed_object_ids`; `remaining_object_ids` must match
every frame's ordered PBR layers exactly and must be disjoint from all removed
objects. This prevents a removed foreground object from being rendered back into
a later clean plate.

## Round kinds and the no-measured terminal contract

`object_peel` always requires
`measured_donor_contract.role=calibrated_rgbd_donor_prefill`. Generated RGB,
ProPainter RGB, unresolved RGB, or a structural render cannot be relabeled as a
measured donor.

`final_background` requires `remaining_object_ids=[]` and a complete immediately
preceding round. It may use the calibrated donor contract. It may instead use the
narrow terminal contract below when no calibrated measured donor exists:

```json
{
  "role": "no_measured_donor_evidence",
  "claims_original_observed_rgb_donor": false,
  "measured_pixels": 0,
  "generated_pixels": 0,
  "propainter_pixels": 0
}
```

This role is evidence of absence, not donor evidence. Every frame must omit
`donor_prefill_rgb`, provide an all-zero `donor_measured_mask`, provide an all-NaN
`donor_measured_depth`, contain no downstream object layers, and provide the
cumulative removal mask. The accepted structural background must cover every
pixel in that removal mask with positive finite depth and sufficient alpha;
partial coverage is rejected rather than left unresolved.

The structural render receipt must have `acceptance_status=accepted`,
`promotion_approved=true`, and a hashed `accepted_texture_report`. That report
must have `status=accepted_for_round04_clean_plate`,
`eligible_as_round04_clean_plate=true`, and both
`acceptance_gates.visual_quality_passed=true` and
`acceptance_gates.technical_quality_passed=true`. Merely self-signing an RGBA
and depth pair is insufficient. No current Bedroom4 candidate satisfies this
contract.

### Limited current-demo acceptance

A separate `accepted_with_limitations` branch exists for an explicitly scoped
demo. It is not strict acceptance and cannot claim unconditional promotion. The
structural receipt must use:

```json
{
  "acceptance_status": "accepted_with_limitations",
  "promotion_approved": false,
  "accepted_with_limitations": true,
  "demo_use_approved": true,
  "eligible_as_round04_clean_plate": false,
  "eligible_as_current_demo_round04_clean_plate": true,
  "acceptance_scope": "current_demo_only"
}
```

Its hashed texture report still uses
`status=accepted_for_round04_clean_plate` and
`eligible_as_round04_clean_plate=false`, while explicitly declaring
`eligible_as_current_demo_round04_clean_plate=true`, `demo_use_approved=true`,
`accepted_with_limitations=true`, `promotion_approved=false`, and
`acceptance_scope=current_demo_only`. Its `acceptance.review_limitations` list
must be non-empty. `acceptance.overridden_gates` is a dictionary containing
exactly `visual_quality`, `wall_boundary_color_continuity`, and
`wall_boundary_low_frequency_gradient_continuity`; `override_gate_keys` must
name the same three gates. The two boundary entries retain numeric threshold,
observed maximum, and `passed=false`; `visual_quality` retains the original
`pending_human_or_vlm_review` state. Top-level `gates` preserves those boundary
metrics and the final `visual_quality=accepted_with_limitations` decision. Every
non-overridden technical gate must remain true.

The receipt must copy `acceptance_scope`, `review_limitations` as `limitations`,
`override_gate_keys`, `overridden_gates`, the two eligibility flags, and demo
approval exactly from the acceptance record. It also records the same override
dictionary as `failed_metrics`; this preserves the observed failed metrics rather
than converting them into a pass. The resulting composite uses
`technical_passed_complete_partition_with_limitations`, propagates
`current_demo_only`, and keeps `promotion_approved=false`. Geometry,
outside-mask exactness, provenance partitioning, shared-atlas consistency, or
any other technical failure cannot be waived by this branch.

## Composition order

1. Copy source RGB.
2. Inside the cumulative removal mask, copy only calibrated measured donor pixels
   and depth. In the explicit no-measured terminal branch this set is empty.
3. Only in the measured residual, z-buffer downstream PBR renders and the
   structural-background render using positive finite depth.
4. Keep unresolved RGB equal to source as a diagnostic; mark it unresolved and
   prohibit promotion.
5. Preserve every RGB pixel outside the cumulative removal mask exactly.

The output labels partition the full frame:

| Label | Class |
| ---: | --- |
| 0 | outside source |
| 1 | measured donor |
| 2 | downstream object render |
| 3 | structural background |
| 4 | unresolved |

Within the removal mask, labels 1 through 4 are mutually exclusive and exactly
exhaustive. Per-class binary masks and a downstream-layer label image are also
written.

## Current blockers

- Scene-fit PBR layers still need camera-aligned RGBA/depth render receipts.
- The planar background still needs a structural RGBA/depth render receipt that
  labels observed, synthetic, or hybrid atlas provenance without claiming the
  measured-donor class, plus an independently hashed accepted texture report.
- A real Bedroom4 composite remains blocked until those inputs exist and visual,
  cross-view, depth, and normal QA pass.
