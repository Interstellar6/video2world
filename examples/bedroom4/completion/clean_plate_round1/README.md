# Bedroom4 clean plate round 1: ProPainter smoke

Status: **REJECTED FOR SCENE PROMOTION**

This evidence package records a real ProPainter run, not a synthetic or proxy result. The
technical execution passed, but visual review rejected the completed pixels. None of these
frames should replace the current Bedroom4 scene or feed PGSR/TSDF reconstruction.

## Reproducibility

- Official source: <https://github.com/sczhou/ProPainter>
- Official paper: <https://arxiv.org/abs/2309.03897> (ICCV 2023)
- Source commit: `e870e79321c31b733e2031af5aa2fb1fe3ac7eec`
- Official weights release: <https://github.com/sczhou/ProPainter/releases/tag/v0.1.0>
- Remote run root: `/root/autodl-tmp/video2world_clean_plate_20260717`
- Runtime: Python 3.10, PyTorch `2.2.2+cu121`, torchvision `0.17.2+cu121`
- Hardware: one NVIDIA RTX 3090, fp16

Required inference weights:

| Weight | Bytes | SHA-256 |
| --- | ---: | --- |
| `ProPainter.pth` | 157,780,510 | `12c070c4b48f374c91d8a2a17851140b85c159621080989f9e191bbc18bd6591` |
| `raft-things.pth` | 21,108,000 | `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1` |
| `recurrent_flow_completion.pth` | 20,348,681 | `22939a1a7900da878dbe1ccd011d646b1bfb30b8290039d8ff0e0c2fefbfd283` |

The official `running_car` smoke completed 292 frames in 45.60 seconds with exit status 0.
Its `inpaint_out.mp4` SHA-256 was
`a85de8fa78b459e4c245c52cf1f015aaf49bedf91de25d4ec37a70171419f9cb`.

## Bedroom4 attempts

| Attempt | Frames | Mask scope | Technical QA | Visual decision |
| --- | ---: | --- | --- | --- |
| `bedroom4_pillow_union_0056_0071` | 16 | three-pillow class union | Passed | Rejected: large brown/gray blur across the headboard and bed |
| `bedroom4_pillow_front_0056_0071` | 16 | tracked front pillow | Passed | Rejected: smaller but obvious gray-white low-frequency patch |
| `bedroom4_pillow_front_0048_0072` | 25 | tracked front pillow, wider camera baseline | Passed | Rejected: patch is more stable but bedding seams and texture are not recovered |

The 25-frame candidate is the least-bad attempt and is the artifact copied into this folder.
It tracks the front pillow from anchor `000064/pillow_03.png` using adjacent-frame bbox and
shape continuity. Raw SAM mask names change during the sequence, so filename identity alone
would be incorrect. Maximum adjacent tracking cost was `0.110265`; each mask covered
18,346-23,209 pixels, or 1.99%-2.52% of a 1280x720 frame.

Technical gates for the 25-frame candidate:

- 25/25 output frames and dimensions matched.
- Maximum mean absolute RGB delta outside the four-pixel dilated mask was `0.0`.
- Minimum per-frame mean absolute RGB delta inside the mask was `22.6442`, confirming that
  the masked pixels were actually replaced.
- Wall time was 37.88 seconds, maximum host RSS was 6,239,332 KiB, and exit status was 0.

These gates only validate I/O integrity. They do not certify semantic or geometric quality.

The official inference chain runs RAFT optical flow, recurrent flow completion, image
propagation, and ProPainter's transformer refinement. This makes it a suitable temporally
consistent 2D residual-hole filler, but it does not introduce calibrated-camera geometry or
3D visibility reasoning.

## Why this result is rejected

The front pillow is removed and the two rear pillows remain correctly placed, but the filled
area is a visibly blurred gray-white patch. It does not reconstruct the hidden bed-sheet seams,
rear-pillow texture, or a sharp contact boundary. Increasing the temporal window from 16 to 25
frames did not recover those details because the target surface remains persistently occluded
and ordinary optical flow has no explicit 3D parallax or visibility model.

## Required next attempt

1. Use DA3 depth plus calibrated cameras to reproject unmasked bed/headboard pixels from
   neighboring views into each target frame.
2. Apply z-buffer and foreground-mask visibility tests, then confidence-fuse the valid donor
   colors into a geometry-aware prefill.
3. Run ProPainter only on residual holes, not on the full object silhouette.
4. Rerun depth/normal estimation on the accepted clean plates before rebuilding PGSR and TSDF;
   reusing foreground depth would recreate the removed pillow geometry.
5. Require multi-view/VLM review and scene-level penetration checks before promotion.

The representative sequence files map as follows: `0000` -> source frame `000048`, `0016` ->
`000064`, and `0024` -> `000072`.

## Artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| `input_manifest.json` | `bd83259bd9c3bd335f8093f68b0515e82a7bc642b0ead27a960891ed33fabd83` |
| `clean_plate_qa.json` | `5e796cbf413f17bfcfa90c029f592942a82522c58ff46d40581ce125acab5f63` |
| `source_mask_output_contact_sheet.png` | `79dde242c6b7fa84238a8ef9330545756eb6a03f2f154de6b0ec21a07f1159e0` |
| `inpaint_out.mp4` | `6db53f516a8ded9be00f294e559a82489f4ead1175137d260e9123f44a1c0830` |
