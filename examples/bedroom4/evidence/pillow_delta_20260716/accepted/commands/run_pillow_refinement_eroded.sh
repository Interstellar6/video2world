#!/usr/bin/env bash
set -euo pipefail

BASE="/data/design/zyx/workspace/holi_spatial_runs/bedroom_4_fresh_da3_sam3_pgsr_20260714_184217"
RAW="/data/design/zyx/workspace/video2world_runs/bedroom_4_pillow_delta_20260716"
REFINE="$RAW/refinements/top3_score090_erode5_depth005_minvotes2"
SOURCE="/data/zyx/workspace/Video2MeshWorkspace/video2mesh_runs/bedroom_4_scene_only_v2mw_20260709_030359"
CODE="$BASE/code_snapshot"
PYTHON="/data/zyx/workspace/pgsr_env/bin/python"
POINT_CLOUD="$BASE/scannetppv2/data/bedroom_4/pointcloud_da3.ply"
STAGING="/data/design/zyx/workspace/video2world_runs/.pillow_delta_20260716_staging"

if [[ -e "$REFINE" ]]; then
  echo "Refusing to reuse existing refinement path: $REFINE" >&2
  exit 2
fi

"$PYTHON" "$CODE/tools/prepare_holi_spatial_bedroom4_full_run.py" \
  --source-project "$SOURCE" \
  --run-root "$REFINE" \
  --scene bedroom_4

mkdir -p "$REFINE/commands" "$REFINE/prompts" "$REFINE/logs" "$REFINE/evidence"
cp "$0" "$REFINE/commands/run_pillow_refinement_eroded.sh"
for script in filter_sam3_mask_index.py erode_class_masks.py build_pillow_delta_report.py verify_pillow_projection.py; do
  cp "$STAGING/$script" "$REFINE/commands/$script"
done
cp "$RAW/prompts/pillow_prompts.json" "$REFINE/prompts/pillow_prompts.json"

exec > >(tee -a "$REFINE/logs/pillow_refinement.log") 2>&1
echo "Started: $(date --iso-8601=seconds)"

"$PYTHON" "$REFINE/commands/filter_sam3_mask_index.py" \
  --input "$RAW/sam3_masks/bedroom_4/mask_index.json" \
  --output "$REFINE/sam3_masks/bedroom_4/mask_index.json" \
  --min-score 0.90 \
  --max-per-frame 3

env PYTHONPATH="$CODE" "$PYTHON" -B \
  "$CODE/tools/holi_spatial_sam3_adapter.py" convert-masks \
  --mask-index "$REFINE/sam3_masks/bedroom_4/mask_index.json" \
  --project-root "$REFINE/video2mesh" \
  --overwrite

"$PYTHON" "$REFINE/commands/erode_class_masks.py" \
  --input-root "$REFINE/video2mesh/masks/2d" \
  --output-root "$REFINE/video2mesh/masks/2d_eroded5" \
  --kernel-size 5

env PYTHONPATH="$CODE" "$PYTHON" -B -m video2mesh.cli fuse-masks \
  --project-root "$REFINE/video2mesh" \
  --mask-root "$REFINE/video2mesh/masks/2d_eroded5" \
  --point-cloud "$POINT_CLOUD" \
  --fusion-mode probability \
  --min-votes 2 \
  --depth-tolerance 0.02 \
  --relative-depth-tolerance 0.005

env PYTHONPATH="$CODE" "$PYTHON" -B \
  "$CODE/tools/postprocess_holi_spatial_3d_instances.py" \
  --project-root "$REFINE/video2mesh" \
  --class-object-masks "$REFINE/video2mesh/masks/3d/object_masks.json" \
  --groundingdino-prompts "$REFINE/prompts/pillow_prompts.json" \
  --max-instances-per-category 3 \
  --min-cluster-fraction 0.01 \
  --holi-bbox-output "$REFINE/output_scannetppv2_new/bedroom_4.json" \
  --overwrite

env PYTHONPATH="$CODE" "$PYTHON" -B -m video2mesh.cli export-object-mask-clouds \
  --project-root "$REFINE/video2mesh" \
  --point-cloud "$POINT_CLOUD" \
  --skip-missing

env PYTHONPATH="$CODE" "$PYTHON" -B \
  "$REFINE/commands/build_pillow_delta_report.py" \
  --delta-root "$REFINE" \
  --base-root "$BASE" \
  --run-id bedroom_4_pillow_top3_score090_erode5_depth005_minvotes2

env PYTHONPATH="$CODE" "$PYTHON" -B \
  "$REFINE/commands/verify_pillow_projection.py" \
  --delta-root "$REFINE" \
  --base-root "$BASE" \
  --min-point-hit-ratio 0.95

sha256sum \
  "$REFINE/sam3_masks/bedroom_4/mask_index.json" \
  "$REFINE/video2mesh/masks/2d_eroded5/erosion_manifest.json" \
  "$REFINE/video2mesh/masks/3d/object_masks.json" \
  "$REFINE/video2mesh/masks/3d_instances/object_masks.json" \
  "$REFINE/output_scannetppv2_new/bedroom_4.json" \
  "$REFINE/evidence/pillow_delta_manifest.json" \
  "$REFINE/evidence/pillow_delta_summary.json" \
  "$REFINE/evidence/projection_verification/projection_verification.json" \
  > "$REFINE/evidence/sha256sums.txt"

echo "Completed: $(date --iso-8601=seconds)"
