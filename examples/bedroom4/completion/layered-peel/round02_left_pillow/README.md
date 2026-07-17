# Round 2: left rear pillow

This round consumes the Round 1 front-pillow peel, completes the left pillow as
an independent PBR candidate, removes it from the 25-frame sequence, and then
reinspects the remaining scene.

## Decision

- TRELLIS.2 seed 42 is rejected: it has open holes, broken surfaces, and
  disconnected fragments.
- TRELLIS.2 seed 44 is accepted with warnings as the object-asset candidate.
  Backside texture banding and a dark underside remain minor warnings.
- The ProPainter output passes technical pixel gates but is usable only for
  SAM3 reinspection. It is not a scene-ready clean plate.
- Calibrated 3D-anchor association selects `sam3_pillow_right` as the sole next
  layer target.

The authoritative machine-readable summary is `round_receipt.json`. The full
25-frame source and output directories remain on mil8; this repository stores
the compact MP4, manifests, QA reports, associations, and review contact sheets.
