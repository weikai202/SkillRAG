#!/usr/bin/env bash
# Submit 3 model jobs (chained) + 1 summary job at the end.
set -euo pipefail

mkdir -p baseline/logs

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

  JOB_ID=$(echo "${JOB_OUTPUT}" | grep -oP '(?<=<)\d+(?=>)')
  PREV_JOB="${JOB_ID}"
  echo "Submitted: ${model} -> Job ${JOB_ID}"
done

# Step 5: summarize after all models finish
JOB_OUTPUT=$(bsub -J "bl_summary" \
  -q dbeigpu \
  -gpu "num=1:mode=shared:gmem=10000" \
  -n 1 \
  -W 30 \
  -o "baseline/logs/summary.out" \
  -e "baseline/logs/summary.err" \
  -w "done(${PREV_JOB})" \
  "source activate PRSAgent 2>/dev/null || conda activate PRSAgent; cd /home/xue0/SkillRAG; python -m baseline.summarize_paper_results")

JOB_ID=$(echo "${JOB_OUTPUT}" | grep -oP '(?<=<)\d+(?=>)')
echo "Submitted: summary -> Job ${JOB_ID}"

echo ""
echo "Job chain: Llama -> Qwen -> Gemma -> Summary"
echo "Monitor with: bjobs -w"
