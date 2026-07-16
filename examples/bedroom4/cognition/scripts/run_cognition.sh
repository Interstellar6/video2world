#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/root/video2world_cognition_20260716"
MODEL_DIR="/root/autodl-tmp/qwen2.5-vl-3b-instruct"
PYTHON="/data/anaconda3/envs/4dlangsplat/bin/python"
STAGING="/root/video2world_cognition_20260716_staging"
GPU="${GPU:-6}"

if [[ -e "$RUN_ROOT/outputs/cognition_manifest.json" ]]; then
  prior_status="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_ROOT/outputs/cognition_manifest.json")"
  if [[ "$prior_status" == "passed" ]]; then
    echo "Refusing to overwrite completed cognition output" >&2
    exit 2
  fi
  prior_output="$RUN_ROOT/outputs.failed.$(date +%Y%m%d_%H%M%S)"
  mv "$RUN_ROOT/outputs" "$prior_output"
  echo "Preserved non-passing prior output at $prior_output"
fi

mkdir -p "$RUN_ROOT/scripts" "$RUN_ROOT/logs" "$RUN_ROOT/inputs" "$RUN_ROOT/outputs" /root/hf_cache /root/tmp
for script in prepare_cognition_inputs.py validate_qwen_model.py generate_object_cognition.py run_cognition.sh; do
  cp "$STAGING/$script" "$RUN_ROOT/scripts/$script"
done

exec > >(tee -a "$RUN_ROOT/logs/cognition.log") 2>&1
echo "started_at=$(date --iso-8601=seconds)"
echo "gpu=$GPU"
df -h /root /root/autodl-tmp
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits

COMMON_ENV=(
  HF_HOME=/root/hf_cache
  HF_HUB_CACHE=/root/hf_cache/hub
  TRANSFORMERS_CACHE=/root/hf_cache/transformers
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  HF_DATASETS_OFFLINE=1
  TMPDIR=/root/tmp
  PYTHONUNBUFFERED=1
)

env "${COMMON_ENV[@]}" "$PYTHON" "$RUN_ROOT/scripts/prepare_cognition_inputs.py" \
  --output-root "$RUN_ROOT/inputs"

env "${COMMON_ENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$RUN_ROOT/scripts/validate_qwen_model.py" \
  --model-dir "$MODEL_DIR" \
  --output "$RUN_ROOT/model_validation.json"

env "${COMMON_ENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$RUN_ROOT/scripts/generate_object_cognition.py" \
  --model-dir "$MODEL_DIR" \
  --input-manifest "$RUN_ROOT/inputs/input_manifest.json" \
  --output-root "$RUN_ROOT/outputs" \
  --max-new-tokens 900

echo "completed_at=$(date --iso-8601=seconds)"
du -sh "$RUN_ROOT" "$MODEL_DIR" /root/hf_cache 2>/dev/null || true
