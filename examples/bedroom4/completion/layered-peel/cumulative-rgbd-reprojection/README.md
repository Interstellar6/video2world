# Strict cumulative RGB-D reprojection

This evidence set measures what the original 25 RGB-D observations can recover
after peeling the three pillows in strict front-to-back order. It does not use
ProPainter, SDXL, a previous prefill frame, or an unresolved residual as donor
evidence.

| Round | Cumulative removal | Mask pixels | Measured RGB-D | Unresolved |
| --- | --- | ---: | ---: | ---: |
| 1 | front pillow | 532,888 | 69 (0.0129%) | 532,819 (99.9871%) |
| 2 | front + left pillow | 1,233,510 | 9,669 (0.7839%) | 1,223,841 (99.2161%) |
| 3 | front + left + right pillow | 1,636,824 | 12,443 (0.7602%) | 1,624,381 (99.2398%) |

All rounds passed the technical gates: pixels outside the cumulative removal
mask are RGB-exact, residual masks are subsets of removal masks, donor RGB
hashes match the original observations, and every round hashes the immediately
preceding manifest/report. These gates do not make the clean plate promotable.

Each measured donor pixel now also has a fused positive metric depth in the
target camera's +Z convention. The `depth/*.npy` arrays are `float32`, are valid
exactly on `removal - residual`, and are `NaN` everywhere else. Every depth file
is SHA-256-bound by its frame record and the output receipt binds the complete
report. Reports produced before this depth upgrade remain intentionally
ineligible for the layered compositor.

The contact sheets show source RGB, cumulative removal mask, measured donor
prefill, and unresolved residual. Magenta is the cumulative removal area;
orange is the part still unresolved after calibrated reprojection. The measured
fraction is too small to call any round a completed clean plate.

The remaining RGB requires one cross-view-consistent background surface and
texture atlas. After final RGB acceptance, depth and normals must be estimated
again before PGSR/TSDF reconstruction. Interactive replacement remains a
separate gate that depends on accepted scene-fit PBR GLBs and
support/interpenetration QA.

See `summary.json` for exact metrics and SHA-256 values. Each round keeps only
small local evidence: cumulative masks, manifest/index receipts, the full
metric report, its receipt, and one contact sheet. Full prefill frames,
residual masks, support maps, sparse fused depth, original RGB, and DA3 donor
depth remain on mil8 under:

`/root/autodl-tmp/video2world_layered_bedroom4_20260717/cumulative_clean_plate_rgbd_v1`
