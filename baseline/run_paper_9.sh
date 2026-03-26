#!/usr/bin/env bash
set -uo pipefail

# Do NOT set CUDA_VISIBLE_DEVICES here — LSF manages GPU allocation via bsub.

MODELS=(
  "NousResearch/Meta-Llama-3-8B-Instruct"
  "Qwen/Qwen3-8B"
  "unsloth/gemma-2-9b-it"
)

for model in "${MODELS[@]}"; do
  echo "====== Model: ${model} ======"
  bash baseline/run_dev500.sh "${model}"
done
