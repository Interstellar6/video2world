# Bed anchor projection supplement

This directory records the depth-visible projection of
`sam3_bed_01.ply` into Bedroom4 frames `000048` through `000072`. The
supplement is intended to be unioned with the existing Round 4 bed-removal
mask so the white bedding and bed front edge omitted by SAM3 are still removed.
It is not an amodal bed reconstruction.

## Version decision

| Version | Status | Decision |
| --- | --- | --- |
| `v1-rejected` | Rejected | Neighbor minimum-depth propagation removed original visible anchor evidence. Frame 000048 fell from 171,717 raw pixels to 73,465 final pixels. Never consume this version. |
| `v2-authoritative` | Technical pass | Splat and close are monotonic supplements: `visible_raw` is a subset of `final_mask`, which is a subset of the depth-consistent anchor envelope, for all 25 frames. |

V2 uses the original PLY anchor, `world_to_camera` calibration, and DA3 depth.
Every added pixel must have propagated anchor depth consistent with observed
DA3 depth. No SAM mask dilation is performed.

## V2 evidence

- Manifest SHA-256: `fbfe05b23d0c7b0a38b06dc64acf51915ff09b8ae06a9b3ca8260d86329d261d`
- Receipt SHA-256: `34a17d36270d33087785af71c6a264358e89d805bf0343e21eb98e485f565460`
- Contact sheet SHA-256: `ed866e9e7ba62a6e2995c91b3a0b67ae2ca5207fa7c2513c0d337eac6363ac0d`
- Mask-set SHA-256: `276ccbff4c8ac5f8f88cec30febb6dba0fcf09d65f383b62d8a9ef52c3292e55`
- Final pixels across 25 frames: 4,499,956
- Per-frame final pixels: minimum 160,694; median 173,002; maximum 209,049

The repo keeps representative masks for frames 000048, 000064, 000067, and
000072. The complete 25-frame mask set and diagnostics remain at:

`/root/autodl-tmp/video2world_layered_bedroom4_20260717/round04_bed/anchor_projection_supplement_v2`

The local working copy used by the planar-background run is:

`/private/tmp/video2world-planar-background/bed-anchor-v2`
