#!/usr/bin/env bash
# Submit 3 model jobs sequentially (each gets 4 GPUs).
# Jobs run one after another via bsub -w dependency.
set -euo pipefail

MODELS=(
  "NousResearch/Meta-Llama-3-8B-Instruct"
  "Qwen/Qwen3-8B"
  "unsloth/gemma-2-9b-it"
)

PREV_JOB=""
for model in "${MODELS[@]}"; do
  SHORT="${model##*/}"
  SHORT="${SHORT//./_}"

  DEP_FLAG=""
  if [ -n "${PREV_JOB}" ]; then
    DEP_FLAG="-w done(${PREV_JOB})"
  fi

  # shellcheck disable=SC2086
  JOB_OUTPUT=$(bsub -J "bl_${SHORT}" \
    -q dbeigpu \
    -gpu "num=4:mode=shared:gmem=120000" \
    -n 4 \
    -W 720 \
    -o "baseline/logs/${SHORT}.out" \
    -e "baseline/logs/${SHORT}.err" \
    ${DEP_FLAG} \
    "source activate PRSAgent 2>/dev/null || conda activate PRSAgent; cd /home/xue0/SkillRAG; bash baseline/run_model.sh '${model}'")

  # Extract job ID
  JOB_ID=$(echo "${JOB_OUTPUT}" | grep -oP '(?<=<)\d+(?=>)')
  PREV_JOB="${JOB_ID}"
  echo "Submitted: ${model} -> Job ${JOB_ID}"
done

echo ""
echo "All 3 jobs submitted in chain. Monitor with: bjobs -w"
