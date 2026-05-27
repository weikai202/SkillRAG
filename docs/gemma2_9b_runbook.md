# Gemma-2-9B SkillRAG Runbook

This runbook is for running SkillRAG with `google/gemma-2-9b-it` using sparse retrieval, CoT prompting, hidden-state prober training, SkillRAG evaluation, and FlashAttention when the local CUDA environment supports it.

## 1. Repository Entry Points

Run all commands from the repository root:

```bash
cd /path/to/SkillRAG
```

Main files:

- `configs/gemma2_9b.yaml`: default Gemma-2-9B experiment config.
- `run_pipeline.py`: end-to-end runner for index building, data generation, prober training, checkpoint checking, evaluation, and reporting.
- `exp_rag.py`: main RAG runner for `none`, `simple`, `probing`, and `skillrag`.
- `train.py`: layer-wise prober training.
- `make_indexer.py`: sparse BM25 or dense FAISS index construction.
- `prompts.py`: CoT, retrieval, router, and skill prompts.
- `utils.py`: shared model loading, FlashAttention fallback, Qwen/Gemma prompt formatting, prober utilities, and metrics helpers.

Important output directories:

- `raw_data/`: downloaded datasets and retrieval corpora.
- `raw_data/sparse_index/`: BM25 index files built by `make_indexer.py`.
- `dataset/9b/`: generated Gemma-2-9B train/dev/evaluation CSV files.
- `ckpt/_3/<dataset>/`: trained `resid_post` prober checkpoints.
- `pckpt/_3/<dataset>/`: trained `resid_mid` prober checkpoints.
- `result/`: evaluation metrics, router counts, hard-case traces, and per-question retrieval/skill stats.
- `reports/`: final YAML reports emitted by `run_pipeline.py`.
- `cache/`: local Hugging Face / TransformerLens model cache when used by `train.py`.

## 2. Environment Setup

Use the project setup script first:

```bash
bash setup_probing_env.sh
conda activate probing
```

If you are using an existing cluster environment, activate it and confirm it has CUDA-enabled PyTorch:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

Gemma-2-9B should be run on a GPU with enough memory for model loading plus generation/prober caching. The current config assumes `cuda:0`.

## 3. FlashAttention Installation

The code now attempts to load models with:

```yaml
attention:
  attn_implementation: flash_attention_2
  dtype: bfloat16
```

This is already present in `configs/gemma2_9b.yaml`. If FlashAttention is unavailable, the loader prints a fallback warning and retries with the default attention backend.

Install build helpers:

```bash
pip install -U packaging psutil ninja
ninja --version
```

Install FlashAttention:

```bash
pip install flash-attn --no-build-isolation
```

On shared machines with limited CPU memory during compilation, limit parallel build jobs:

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

Verify installation:

```bash
python - <<'PY'
import torch
import flash_attn
from transformers.utils import is_flash_attn_2_available
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("flash_attn:", getattr(flash_attn, "__version__", "installed"))
print("hf flash_attn_2 available:", is_flash_attn_2_available())
PY
```

Expected model-loading log when FlashAttention is active:

```text
[attention] using attn_implementation=flash_attention_2, dtype=bfloat16
```

Fallback log when unsupported:

```text
[attention] flash_attention_2 unavailable (...); retrying without flash attention.
```

To disable FlashAttention for debugging, either edit the config:

```yaml
attention:
  attn_implementation: default
  dtype: bfloat16
```

or pass CLI overrides when running individual scripts:

```bash
python exp_rag.py ... --attn_implementation default --dtype bfloat16
python train.py ... --attn_implementation default --dtype bfloat16
```

## 4. Data Preparation

Download required datasets:

```bash
bash download/download.sh
bash download/raw_data.sh
```

Quick checks:

```bash
ls raw_data/nq/biencoder-nq-train.json
ls raw_data/nq/biencoder-nq-dev.json
ls raw_data/hotpotqa/hotpot_train_v1.1.json
ls raw_data/trivia/biencoder-trivia-train.json
```

The Gemma-2-9B config builds sparse indexes for:

```text
nq, musique, hotpotqa, trivia, 2wikimultihopqa
```

You can build one index manually:

```bash
python make_indexer.py --dataset_name nq --is_sparse
```

## 5. Full Pipeline

Dry-run first:

```bash
python run_pipeline.py --config configs/gemma2_9b.yaml --dry_run
```

Run the full experiment:

```bash
python run_pipeline.py --config configs/gemma2_9b.yaml
```

Resume from a previous report:

```bash
python run_pipeline.py \
  --config configs/gemma2_9b.yaml \
  --resume_from reports/gemma-2-9b-it_YYYYMMDD_HHMMSS.yaml
```

## 6. Default Experiment Parameters

From `configs/gemma2_9b.yaml`:

```yaml
model:
  id: google/gemma-2-9b-it

attention:
  attn_implementation: flash_attention_2
  dtype: bfloat16

index_datasets: [nq, musique, hotpotqa, trivia, 2wikimultihopqa]

retrieval:
  is_sparse: true
  is_cot: true

build_dataset:
  datasets: [nq, hotpotqa, trivia]
  steps_limit_train: 3000
  steps_limit_dev: 500
  sep_number: 0

train:
  method: tokens_mean
  batch_size: 6
  lr: 0.001
  epochs: 2
  layers: [12, 16, 20, 24, 28, 32, 36, 40]
  device: cuda:0
  train_ds_ratio: 1.0
  disable_wandb: true

prober:
  ds: 3
  ablation: 0
  ckpt_dataset: nq

evaluate:
  datasets: [nq, musique, hotpotqa, trivia, 2wikimultihopqa]
  methods: [none, simple, probing, skillrag]
  steps_limit: 500
  threshold: 0.0
  position: resid_post
  max_retrieval_rounds: 3
  tr_or_dev: dev
  extracting_cot_qa: true
  extract_sep: true
  sep_number: 0
```

Interpretation:

- `steps_limit_train: 3000`: generate about 3000 train examples per build dataset for `none` and `simple`.
- `steps_limit_dev: 500`: generate about 500 dev examples per build dataset.
- `train.layers`: train one prober per listed Gemma-2-9B layer.
- `prober.ds: 3`: checkpoint namespace used under `ckpt/_3/` and `pckpt/_3/`.
- `evaluate.methods`: compare no retrieval, one-shot retrieval, prober-gated retrieval, and SkillRAG.
- `evaluate.max_retrieval_rounds: 3`: maximum retrieval rounds for `probing` and `skillrag`.
- `evaluate.threshold: 0.0`: prober decision margin used during evaluation.
- `attention.dtype: bfloat16`: lowers model activation memory; probe inputs are cast back to fp32 before probe classifiers.

## 7. Running Individual Stages

Build Gemma-2-9B prober construction data for one dataset:

```bash
python exp_rag.py \
  --retr_method none \
  --is_sparse \
  --tr_or_dev train \
  --extracting_cot_qa \
  --extract_sep \
  --steps_limit 3000 \
  --dataset_name nq \
  --is_cot \
  --sep_number 0 \
  --model_id google/gemma-2-9b-it \
  --attn_implementation flash_attention_2 \
  --dtype bfloat16

python exp_rag.py \
  --retr_method simple \
  --is_sparse \
  --tr_or_dev train \
  --extracting_cot_qa \
  --extract_sep \
  --steps_limit 3000 \
  --dataset_name nq \
  --is_cot \
  --sep_number 0 \
  --model_id google/gemma-2-9b-it \
  --attn_implementation flash_attention_2 \
  --dtype bfloat16
```

Balance one training dataset:

```bash
python balance_train_dataset.py \
  --model_id google/gemma-2-9b-it \
  --dataset_name nq
```

Train one prober layer:

```bash
python train.py \
  --method tokens_mean \
  --batch_size 6 \
  --lr 0.001 \
  --layer 24 \
  --device cuda:0 \
  --epochs 2 \
  --model_id google/gemma-2-9b-it \
  --dataset_name nq \
  --train_ds_ratio 1.0 \
  --attn_implementation flash_attention_2 \
  --dtype bfloat16 \
  --disable_wandb
```

Evaluate SkillRAG on one dataset:

```bash
python exp_rag.py \
  --retr_method skillrag \
  --steps_limit 500 \
  --dataset_name nq \
  --tr_or_dev dev \
  --is_cot \
  --is_sparse \
  --model_id google/gemma-2-9b-it \
  --ds 3 \
  --position resid_post \
  --threshold 0.0 \
  --max_retrieval_rounds 3 \
  --extracting_cot_qa \
  --extract_sep \
  --sep_number 0 \
  --prober_train_dataset nq \
  --attn_implementation flash_attention_2 \
  --dtype bfloat16
```

## 8. Output Files

Generated train/dev data:

```text
dataset/9b/retrieval_qa_gemma-2-9b-it_<dataset>_<method>_<split>_after0_<steps>.csv
dataset/9b/retrieval_qa_gemma-2-9b-it_<dataset>_all_train_in3_.csv
dataset/9b/retrieval_qa_gemma-2-9b-it_<dataset>_all_train_in3_balanced.csv
dataset/9b/retrieval_qa_gemma-2-9b-it_<dataset>_all_zeroshot_test_500.csv
```

Probe checkpoints:

```text
ckpt/_3/<dataset>/in3_1.0_gemma-2-9b-it_tokens_mean_2_l<layer>_resid_post_ep<epoch>.pt
pckpt/_3/<dataset>/in3_1.0_gemma-2-9b-it_tokens_mean_2_l<layer>_resid_mid_ep<epoch>.pt
```

Evaluation and analysis:

```text
result/*.csv
result/metrics_log.yaml
result/skill_router_counts.csv
result/question_retrieval_skill_stats.csv
reports/gemma-2-9b-it_*.yaml
```

Important analysis files:

- `result/skill_router_counts.csv`: one row per SkillRAG run with total counts and ratios for `query_misaligned`, `multi_hop_missing`, `evidence_not_used`, and `insufficient_evidence`.
- `result/question_retrieval_skill_stats.csv`: one row per question with retrieval rounds, per-question skill counts, selected skill sequence, accuracy, and exit-skill metadata.
- `reports/gemma-2-9b-it_*.yaml`: final pipeline report with build metrics, evaluation metrics, SkillRAG traces, command logs, and grouped retrieval/skill statistics.

## 9. Inspecting Results

Aggregate retrieval rounds and skill calls by dataset/method:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("result/question_retrieval_skill_stats.csv")
df = df[df["model_id"] == "google/gemma-2-9b-it"]
cols = [
    "retrieval_rounds",
    "query_misaligned_count",
    "multi_hop_missing_count",
    "evidence_not_used_count",
    "insufficient_evidence_count",
]
print(df.groupby(["dataset_name", "retr_method"])[cols].agg(["mean", "sum", "max"]))
PY
```

Inspect SkillRAG router distribution:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("result/skill_router_counts.csv")
df = df[df["model_id"] == "google/gemma-2-9b-it"]
cols = [
    "query_misaligned_count",
    "multi_hop_missing_count",
    "evidence_not_used_count",
    "insufficient_evidence_count",
]
print(df.groupby(["dataset_name", "tr_or_dev"])[cols].sum())
PY
```

Inspect whether the `insufficient_evidence` exit skill saved retrieval rounds:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("result/question_retrieval_skill_stats.csv")
df = df[(df["model_id"] == "google/gemma-2-9b-it") & (df["retr_method"] == "skillrag")]
print("exit rows:", int(df["stopped_by_exit_skill"].sum()))
print("saved rounds:", int(df["exit_saved_rounds"].sum()))
print("estimated saved new tokens:", int(df["exit_est_saved_new_tokens"].sum()))
PY
```

## 10. Common Issues

FlashAttention install fails:

- Confirm CUDA-enabled PyTorch works before installing `flash-attn`.
- Reinstall `ninja` if `ninja --version` fails.
- Retry with fewer compile jobs: `MAX_JOBS=4 pip install flash-attn --no-build-isolation`.
- If the cluster image cannot compile FlashAttention, keep `attn_implementation: default`; the code will still run with the default attention backend.

Out of memory during prober training:

- Reduce `train.batch_size`.
- Train fewer layers at once by editing `train.layers`.
- If needed, add `train.max_length` to the config and pass it through `run_pipeline.py`.

Missing prober checkpoints:

```bash
python check_prober_ckpt.py \
  --model_id google/gemma-2-9b-it \
  --layers 12,16,20,24,28,32,36,40 \
  --ds 3 \
  --epoch 1 \
  --dataset_name nq
```

Missing sparse index:

```bash
python make_indexer.py --dataset_name <dataset> --is_sparse
```

Hugging Face gated model access:

- Make sure the environment has a valid Hugging Face token if your cluster requires authenticated model downloads.
- Verify model loading with a short dry-run before launching the full pipeline.
