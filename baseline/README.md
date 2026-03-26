# Baseline

This directory is a self-contained baseline subsystem.

Constraints:

- All new code lives under `baseline/` only.
- Data is downloaded to `baseline/raw_data/`.
- BM25 indexes are written to `baseline/raw_data/sparse_index/`.
- Outputs go to `baseline/results/`.
- No changes to existing files in the repository root.

## Features

- Download five datasets: `hotpotqa`, `2wikimultihopqa`, `musique`, `nq`, `trivia`.
- Build BM25 indexes using names compatible with the main project:
  `baseline/raw_data/sparse_index/llama_index_bm25_model_{dataset}_2.json`
- Run three BM25 baselines:
  - `FLARE`
  - `DRAGIN`
  - `Adaptive-RAG`

## Setup

```bash
pip install -r baseline/requirements.txt
```

## Download data

```bash
bash baseline/download.sh
```

Or:

```bash
python -m baseline.download_data --datasets all
```

## Build BM25 indexes

```bash
bash baseline/make_index.sh
```

Or a single dataset:

```bash
python -m baseline.build_bm25_index --dataset_name hotpotqa
```

## Run a single baseline

```bash
python -m baseline.run_baseline \
  --method flare \
  --dataset_name hotpotqa \
  --model_id google/gemma-2-9b-it \
  --split dev \
  --limit 500
```

## Run all three baselines on all five datasets

```bash
bash baseline/run_dev500.sh google/gemma-2-9b-it
```

Run all three paper models in one shot:

```bash
bash baseline/run_paper_9.sh
```

Models:

- `meta-llama/Meta-Llama-3-8B-Instruct`
- `Qwen/Qwen3-8B`
- `google/gemma-2-9b-it`

After runs, build the paper-style baseline summary table:

```bash
python -m baseline.summarize_paper_results
```

## Outputs

- Per-example: `baseline/results/<model_name>/<method>_<dataset>_<split>_<limit>.csv`
- Run summary: `baseline/results/<model_name>/<method>_<dataset>_<split>_<limit>.summary.json`
- Paper rollup: `baseline/results/paper_baseline_summary.csv`

## Notes

These three methods are BM25-compatible implementations tailored to this project:

- `FLARE`: low-confidence drafts trigger query reformulation plus retrieval.
- `DRAGIN`: high-entropy drafts trigger retrieval and iterative correction.
- `Adaptive-RAG`: route first to no-retrieval / single-hop / multi-hop, then run the matching strategy.

The goal is to satisfy the engineering need to run baselines with the current index format.

This is not a line-by-line port of the official FLARE / DRAGIN / Adaptive-RAG codebases; it is a BM25-compatible implementation. If the final paper must claim strict reproduction of the original baselines, you will need to swap in the official router, query generator, and stopping logic inside `baseline/`, or wire up the upstream implementations directly.
