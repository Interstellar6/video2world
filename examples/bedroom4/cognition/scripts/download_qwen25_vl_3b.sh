#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/root/video2world_cognition_20260716"
MODEL_DIR="/root/autodl-tmp/qwen2.5-vl-3b-instruct"
HF_HOME="/root/hf_cache"
HF_HUB_CACHE="$HF_HOME/hub"
TMPDIR="/root/tmp"
HF_ENDPOINT="https://hf-mirror.com"
HF_BIN="/data/anaconda3/envs/4dlangsplat/bin/hf"
REVISION="66285546d2b821cf421d4f5eb2576359d3770cd3"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/scripts" "$MODEL_DIR" "$HF_HUB_CACHE" "$TMPDIR"
cp "$0" "$RUN_ROOT/scripts/download_qwen25_vl_3b.sh"

exec > >(tee -a "$RUN_ROOT/logs/model_download.log") 2>&1

echo "started_at=$(date --iso-8601=seconds)"
echo "repo=Qwen/Qwen2.5-VL-3B-Instruct"
echo "revision=$REVISION"
echo "model_dir=$MODEL_DIR"
echo "hf_home=$HF_HOME"
echo "hf_hub_cache=$HF_HUB_CACHE"
echo "tmpdir=$TMPDIR"
echo "hf_endpoint=$HF_ENDPOINT"
echo "network_turbo=/etc/network_turbo unavailable on this host; using resumable hf download via configured mirror"
df -h /root /root/autodl-tmp
du -sh "$MODEL_DIR" "$HF_HOME" 2>/dev/null || true

env \
  HF_HOME="$HF_HOME" \
  HF_HUB_CACHE="$HF_HUB_CACHE" \
  HF_ENDPOINT="$HF_ENDPOINT" \
  TMPDIR="$TMPDIR" \
  PYTHONUNBUFFERED=1 \
  "$HF_BIN" download Qwen/Qwen2.5-VL-3B-Instruct \
    --revision "$REVISION" \
    --cache-dir "$HF_HUB_CACHE" \
    --local-dir "$MODEL_DIR" \
    --max-workers 4

echo "completed_at=$(date --iso-8601=seconds)"
df -h /root /root/autodl-tmp
du -sh "$MODEL_DIR" "$HF_HOME" 2>/dev/null || true
