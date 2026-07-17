# Bedroom4 clean plate round 2: calibrated donor reprojection

Status: **REJECTED FOR SCENE PROMOTION**

This package records a real calibrated multi-view attempt on Bedroom4. It is not a
synthetic clean plate. The geometry and I/O gates passed, but the video does not expose
enough of the surface hidden by the front white pillow to repair it from observed pixels.
None of these frames should replace the scene or feed PGSR/TSDF reconstruction.

![Conservative calibrated donor result](multiview_prefill_contact_sheet.png)

## Method

`scripts/prefill_clean_plate_multiview.py` uses the real 80-frame Bedroom4 RGB sequence,
DA3 depth arrays, and `world_to_camera` calibration. For every target frame it:

1. excludes all SAM3 `pillow` pixels in each donor frame;
2. lifts the remaining donor RGB-D pixels to world space;
3. projects them into the target camera and keeps the nearest sample per donor pixel;
4. retains the median-depth cluster supported by at least two donors;
5. writes only accepted colors inside the removal mask and preserves all outside pixels
   exactly;
6. emits the remaining hole as a ProPainter-compatible residual mask.

The recorded DA3 depth is **reprojection evidence only**. Even if an RGB clean plate is
accepted later, depth and normals must be re-estimated from the clean frames before PGSR or
TSDF. Reusing these depth arrays would restore the removed pillow geometry.

## Real attempts

| Attempt | Pillow exclusion | Min support | Frame 48 | Frame 64 | Frame 72 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `representative_v1` | 6 px | 2 | 0.00% | 0.00% | 0.00% | Too conservative; no observed donor support |
| `representative_v2` | 2 px | 2 | 0.62% | 0.17% | 0.59% | Valid but far too sparse for clean-plate recovery |
| `representative_v3_zero_margin` | 0 px | 3 | 11.13% | 9.86% | 11.89% | Rejected; support is a pillow-edge ring |

The conservative attempt changed only 113/18,346, 40/23,065, and 127/21,440 removal-mask
pixels. The residual masks therefore retain 99.38%, 99.83%, and 99.41% of the original holes.
Running ProPainter on those residuals would effectively repeat the already rejected Round 1
run, which produced a gray low-frequency patch instead of bedding seams and texture.

The zero-margin diagnostic is deliberately retained as a failure control. Its support map is
concentrated around the pillow silhouette:

![Zero-margin support is an edge ring](frame64_zero_margin_support.png)

That extra coverage comes from SAM boundary leakage and copies the white pillow edge. It is
not evidence of the bed surface hidden behind or under the pillow. Promoting it would preserve
the exact foreground contamination that the clean-plate stage is meant to remove.

## Visual review

The source frames consistently show that the target front pillow is white or light gray. The
conservative prefill leaves essentially the full white pillow because the hidden bed/headboard
surface is never directly observed by this camera path. The result fails semantic texture
continuity and object-removal completeness even though outside-mask preservation is exact.

The next useful approach needs a generative, geometry-conditioned background prior, such as a
bed/headboard plane-aware texture completion model or a novel-view scene inpainting model. It
must then pass cross-view review before new depth/normal estimation and PGSR/TSDF rebuilding.
Simple donor reprojection plus residual ProPainter is insufficient for this permanently
occluded region.

## Provenance

- Remote run root: `/root/autodl-tmp/video2world_clean_plate_multiview_20260717`
- RGB frames: `bedroom_4_scene_only_v2mw_20260709_030359/scene/frames`
- Camera calibration SHA-256:
  `2901497f144c7378961ad6116a48474dfec475acd35a7a80258db18d36b87568`
- Round 1 input manifest SHA-256:
  `bd83259bd9c3bd335f8093f68b0515e82a7bc642b0ead27a960891ed33fabd83`
- SAM3 mask index SHA-256:
  `5de1e75c9252c8cdc429238209f4a9546c35ee6229b1f19ae4e4eee59956df38`
- Executed remote script SHA-256:
  `295ebf61568347184c4a41d4ea9c5dedd6dce0711874989f53e1cfac9ac9ac1a`
- Current checked-in script SHA-256:
  `6d21e527c4a6eafd335a99c6d70812f4176fe2e0a232c82229e9efa7843238a7`
  (the only later change closes contact-sheet image handles; projection output is unchanged)

Artifact hashes:

| Artifact | SHA-256 |
| --- | --- |
| `multiview_prefill_report.json` | `4b82c8825c96f1ac0e2e296c7e91658e964c08a751c7843ab35e0bcd7289e992` |
| `multiview_prefill_contact_sheet.png` | `e0691b29a0df1d21c7fecf1cb62892e64bb1e03b5b2bba7bbd1cbe165787ccf1` |
| `zero_margin_report.json` | `dc2f260bde6be8b5f7eb9087128e242419edc2324e114d4cb7a8264054072c1b` |
| `zero_margin_contact_sheet.png` | `55727567782d5128e8f464913777578cfd7caeabd94f50c18eebf6122e6ebd0d` |
| `frame64_zero_margin_support.png` | `b8fa37496391f42d6823fc9b7421a7fc75926d504b1fc1fa7bb763a3ed4689b4` |
