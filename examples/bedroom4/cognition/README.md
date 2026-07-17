# Bedroom 4 cognition evidence

This directory is the self-contained evidence bundle for the bedroom 4 open-vocabulary object descriptions.

## Result

- Status: `passed` for all 6 objects.
- Model: `Qwen/Qwen2.5-VL-3B-Instruct` at revision `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Runtime: `transformers 4.57.3`, `torch 2.5.1+cu124`, BF16, SDPA, NVIDIA RTX 3090.
- Verified model size: 3,754,622,976 parameters; 7,509,245,952 tensor-payload bytes.
- Load validation: no missing, unexpected, or mismatched keys.

| Object | Audited bilingual caption (Chinese) |
| --- | --- |
| `sam3_nightstand_01` | 这件素面深暖棕色床头柜呈平滑木质观感，具有矩形柜面、围板和四条方截面直腿，未见抽屉或把手。 |
| `sam3_nightstand_02` | 这件暖红棕色床头柜呈平滑木质观感，正面有一只带深色圆形旋钮的抽屉，并带抬高的弧形后沿与侧沿和四条柜腿。 |
| `sam3_plant_01` | 这株浓密的圆冠造型植物由许多带浅色叶缘的深绿色小椭圆叶、数根木质观感枝干和一个光滑的灰白色圆柱形花盆组成。 |
| `sam3_plant_02` | 这株浓密的圆冠盆栽具有掌状观感的绿奶油色斑锦叶、数根直立枝干和一个光滑的灰白色圆柱形花盆。 |
| `sam3_plant_03` | 这株盆栽具有宽披针形叶片、浅奶油绿色斑锦与深绿色叶缘或条纹、簇生枝干和一个光滑的灰白色圆柱形花盆。 |
| `sam3_pillow_01` | 这个已接受整体由床上三只相接的浅色枕头组成：两只较大的灰白/米白色绗缝花卉枕头位于一只较小的白色/灰白色方枕后方；暖色室内光会使表面略显偏黄。 |

## Consumption

- `outputs/cognition_manifest.json` is the aggregate entry point.
- `outputs/<object_id>.json` contains the final `description`, the untouched parsed `model_description`, quality checks, provenance, prompt/raw hashes, and a schema-compatible `description_evidence` projection.
- `outputs/raw/` preserves the exact local-model text; `outputs/prompts/` preserves the exact prompts.
- `inputs/input_manifest.json` records every evidence source, SHA-256, deterministic image derivation, and run-root-relative path.
- `model_validation.json` records revision evidence, shard/index checks, tensor payload sizes, CUDA load statistics, and loading-key checks.

The final `description` is not presented as untouched model text. It is the local VLM result after explicit schema normalization and pixel-level visual-audit corrections. Every correction is listed in `quality.deterministic_normalizations`; the original remains in `model_description` and `outputs/raw/`.

The two non-passing attempts are preserved under `outputs.failed.*`. They document why schema-only acceptance was insufficient: scalar/list drift, synthetic-background placement claims, exact-material overclaims, and a pillow-count contradiction.

## Evidence boundaries

- Nightstand and plant inputs are the exact EmbodiedGen prompted-reference RGBA images, composited deterministically over a neutral background for the VLM. Their room position is not inferable.
- Pillow evidence is accepted SAM3 frame `000064` plus the eroded mask. The mask has three connected components (28,576, 21,788, and 16,702 pixels) treated as one ensemble without stable per-pillow IDs.
- The original VLM called the rear pillows pale green. Source-frame reinspection corrected the consumable description to white/off-white/cream under warm lighting; the untouched raw model output remains archived for audit.
- Light-gray backgrounds and red mask contours are synthetic derivatives and are excluded from object attributes.
- Scene coordinates use non-metric scene scale. Descriptions do not report meters or physical dimensions.
- Plant species and exact material substrates remain intentionally unspecified.

Model weights remain on mil8 at `/root/autodl-tmp/qwen2.5-vl-3b-instruct` and are not copied into this repository bundle.
