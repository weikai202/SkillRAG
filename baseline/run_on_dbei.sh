#!/usr/bin/env bash
# =============================================================================
# Master script for running baselines on DBEI GPU cluster.
#
# Usage (from the DBEI login node):
#   ssh xue0@scisub9.pmacs.upenn.edu
#   bsub -Is -q dbeigpu -gpu "num=2:mode=shared:gmem=120000" -n 2 'bash'
#   conda activate PRSAgent
#   cd ~/SkillRAG       # or wherever the repo is cloned
#   bash baseline/run_on_dbei.sh
#
# This script does everything: env setup → download → index → run 45 experiments → summarize
# =============================================================================
set -uo pipefail

# --- HuggingFace env (required on DBEI) ---
unset HF_ENDPOINT 2>/dev/null || true
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_ENABLE_XET=0
export TRANSFORMERS_NO_TF_IMPORT=1
export TRANSFORMERS_NO_JAX_IMPORT=1

echo "=== GPU check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

# --- Step 1: Install deps (idempotent) ---
echo "=== Step 1: pip install ==="
pip install -q -r baseline/requirements.txt

# --- Step 2: Download datasets ---
echo "=== Step 2: Download datasets ==="
bash baseline/download.sh

# --- Step 3: Build BM25 indexes ---
echo "=== Step 3: Build BM25 indexes ==="
bash baseline/make_index.sh

# --- Step 4: Run 3 models × 3 methods × 5 datasets = 45 experiments ---
echo "=== Step 4: Run baselines ==="
bash baseline/run_paper_9.sh

# --- Step 5: Summarize ---
echo "=== Step 5: Summarize results ==="
python -m baseline.summarize_paper_results

echo ""
echo "Done. Results at: baseline/results/paper_baseline_summary.csv"
