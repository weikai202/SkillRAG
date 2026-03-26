#!/usr/bin/env bash
# Run all 15 experiments (3 methods × 5 datasets) for a single model.
# Skips experiments whose summary.json already exists.
# Usage: bash baseline/run_model.sh <model_id>
set -uo pipefail

unset HF_ENDPOINT 2>/dev/null || true
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_NO_TF_IMPORT=1
export TRANSFORMERS_NO_JAX_IMPORT=1

MODEL_ID="${1:?Usage: bash baseline/run_model.sh <model_id>}"
# Derive result dir name the same way Python does
MODEL_SHORT="${MODEL_ID##*/}"
MODEL_DIR="baseline/results/${MODEL_SHORT//./_}"

echo "=== GPU check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""
echo "=== Running model: ${MODEL_ID} ==="

FAILED=0
SKIPPED=0
for method in flare dragin adaptive; do
  for dataset in hotpotqa 2wikimultihopqa musique nq trivia; do
    SUMMARY="${MODEL_DIR}/${method}_${dataset}_dev_500.summary.json"
    if [ -f "${SUMMARY}" ]; then
      echo "SKIP (done): ${method} / ${dataset}"
      SKIPPED=$((SKIPPED+1))
      continue
    fi
    echo "=== ${method} / ${dataset} / ${MODEL_ID} ==="
    python -m baseline.run_baseline \
      --method "${method}" \
      --dataset_name "${dataset}" \
      --model_id "${MODEL_ID}" \
      --split dev \
      --limit 500 \
    || { echo "FAILED: ${method} ${dataset} ${MODEL_ID}"; FAILED=$((FAILED+1)); }
  done
done

echo ""
echo "=== Done: ${MODEL_ID} | skipped=${SKIPPED} failed=${FAILED} ==="

# Clean up model cache to free disk quota for next model
CACHE_DIR="${HOME}/.cache/huggingface/hub/models--${MODEL_ID//\//--}"
if [ -d "${CACHE_DIR}" ]; then
  echo "Cleaning cache: ${CACHE_DIR}"
  rm -rf "${CACHE_DIR}"
fi
rm -rf "${HOME}/.cache/huggingface/xet/" 2>/dev/null || true

if [ "${FAILED}" -gt 0 ]; then
  exit 1
fi
