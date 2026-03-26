#!/usr/bin/env bash
set -uo pipefail

MODEL_ID="${1:-google/gemma-2b}"
FAILED=0

for method in flare dragin adaptive; do
  for dataset in hotpotqa 2wikimultihopqa musique nq trivia; do
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

if [ "${FAILED}" -gt 0 ]; then
  echo "${FAILED} run(s) failed."
  exit 1
fi
