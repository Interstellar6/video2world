# Round 4: bed

This round consumes the Round 3 sequence and attempts both independent bed
asset reconstruction and removal of the bed for final background recovery.

## Current state

- The bed input and cavity-aware amodal mask pass their technical gates. The
  bridge is used only to identify the enclosed pillow cavity; no exterior
  bridge pixels are added to the output contour.
- TRELLIS.2 seed 45 is rejected. Its main category and colors are plausible,
  but a large ridge near the headboard and headboard/mattress interpenetration
  fail the unified visual-and-collision gate.
- The 25-frame ProPainter result passes I/O and pixel-delta checks but fails
  visual review: the bed becomes a large blurred field rather than an empty
  room with reconstructed wall and floor.
- The rejected clean-plate candidate is evidence-only and is deliberately not
  exposed as the input to the final-background round.

Round 4 remains `running` with two independent blockers: a new bed asset retry
and geometry-aware final background completion. See `round_receipt.json` for
the exact hashes and decision boundaries.
