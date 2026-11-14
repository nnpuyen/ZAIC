#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_path="${GEMMA4_MODEL_PATH:-$script_dir/models/gemma-4-12B}"

if [[ ! -f "$model_path/config.json" || ! -f "$model_path/model.safetensors" ]]; then
  echo "Gemma 4 checkpoint not found at: $model_path" >&2
  exit 1
fi

exec vllm serve "$model_path" \
  --host 127.0.0.1 \
  --port 8001 \
  --served-model-name gemma4-12b-planner \
  --dtype auto \
  --max-model-len "${GEMMA4_CONTEXT_SIZE:-16384}" \
  --gpu-memory-utilization "${GEMMA4_GPU_MEMORY_UTILIZATION:-0.90}" \
  --limit-mm-per-prompt image=0,audio=0
