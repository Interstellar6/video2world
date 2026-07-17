# Round 3: right rear pillow

This round consumes the Round 2 clean-plate sequence, completes and peels the
right pillow, then runs two separate calibrated association checks: one for
pillow residuals and one for the bed.

## Decision

- TRELLIS.2 seed 43 is accepted with warnings as the right-pillow object-asset
  candidate. The full volume and main light color pass; backside darkness and
  small pattern drift remain warnings.
- The ProPainter output passes technical pixel gates but remains restricted to
  reinspection because the removed region is blurred.
- Pillow association returns no new pillow target. The right pillow is absent
  as expected; front and left residual detections are excluded by peeled-object
  identity.
- Bed association passes in all 25 frames with the recorded bed-specific mask
  hit threshold and selects `sam3_bed_01` for Round 4.

See `round_receipt.json` for content-addressed lineage and exact limitations.
The repository intentionally stores no 25-frame image directory.
