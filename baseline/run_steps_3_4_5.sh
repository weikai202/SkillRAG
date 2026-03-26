#!/usr/bin/env bash
# Steps 3→4→5 only. Step 3 skips datasets whose index already exists.
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

# --- Step 3: Build BM25 indexes (skip if already built) ---
echo "=== Step 3: Build BM25 indexes (incremental) ==="
INDEX_DIR="baseline/raw_data/sparse_index"
for ds in hotpotqa nq trivia 2wikimultihopqa musique; do
  idx_file="${INDEX_DIR}/llama_index_bm25_model_${ds}_2.json"
  if [ -f "$idx_file" ]; then
    echo "SKIP: ${ds} (index already exists: ${idx_file})"
  else
    echo "BUILD: ${ds}"
    python -m baseline.build_bm25_index --dataset_name "$ds"
  fi
done

echo ""
echo "=== All indexes ==="
ls -lh "${INDEX_DIR}"/*.json 2>/dev/null || echo "WARNING: no indexes found"
echo ""

# --- Step 4: Run 3 models × 3 methods × 5 datasets = 45 experiments ---
echo "=== Step 4: Run baselines ==="
bash baseline/run_paper_9.sh

# --- Step 5: Summarize ---
echo "=== Step 5: Summarize results ==="
python -m baseline.summarize_paper_results

echo ""
echo "Done. Results at: baseline/results/paper_baseline_summary.csv"
