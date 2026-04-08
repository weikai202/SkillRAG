# SkillRAG

## Abstract

## Environment Setup
Run the following command in the project root to automatically create and validate the environment:

```bash
bash setup_probing_env.sh
```
This script will:
- create a `conda` environment named `probing` (Python 3.10),
- install all required dependencies,
- download `en_core_web_sm`,
- run a sanity check for key packages and runtime readiness.
## Datasets
Download all required datasets and processed raw files with:
```bash
bash download/download.sh
bash download/raw_data.sh
```

After downloading, you can quickly verify key files:
```bash
ls raw_data/nq/biencoder-nq-train.json
ls raw_data/nq/biencoder-nq-dev.json
```

## Quick Start
Minimal reproducible workflow:

```bash
# 1) environment
bash setup_probing_env.sh

# 2) download data
bash download/download.sh
bash download/raw_data.sh

# 3) verify pipeline commands (no execution)
python run_pipeline.py --config configs/gemma2_9b.yaml --dry_run

# 4) run full pipeline
python run_pipeline.py --config configs/gemma2_9b.yaml
```

## Whole Pipeline
Run full pipeline:
```bash
python run_pipeline.py --config configs/gemma2_9b.yaml
```

Dry-run only (print commands without execution):
```bash
python run_pipeline.py --config configs/gemma2_9b.yaml --dry_run
python run_pipeline.py --config configs/llama3_8b.yaml --dry_run
python run_pipeline.py --config configs/qwen3_8b.yaml --dry_run
```

Current config behavior:
- Build index on 5 datasets: `nq, musique, hotpotqa, trivia, 2wikimultihopqa`
- Build/train prober on 3 datasets: `nq, hotpotqa, trivia`
- Evaluate `none/simple/probing/skillrag` on 5 datasets
- `wandb` disabled in training by default (`--disable_wandb`)

Prober checkpoint selection during evaluation:
- For `nq/hotpotqa/trivia`: pass `--prober_train_dataset` as the same dataset
- For `musique/2wikimultihopqa`: no `--prober_train_dataset` argument (default fallback to `nq`)

Main outputs:
- Prober checkpoints: `ckpt/_3/<dataset_name>/...` and `pckpt/_3/<dataset_name>/...`
- Per-sample generation csv: `dataset/{2b|8b|9b}/retrieval_qa_*.csv`
- Aggregated eval metrics csv: `result/*.csv`
- Run report yaml: `reports/*.yaml`
- Appended metrics log: `result/metrics_log.yaml`








## Creating a BM25 Retrieval Index Based on Llama Index
```bash
bash make_index.sh
```

## Building a Prober Training Dataset

To train the prober, we need to create a dataset using the single-step retrieval method and the no-retrieval method. The code for creating this dataset is provided below.
```bash
bash make_dataset.sh
bash make_dataset_dev.sh
```

## Prober Training
You can train the prober using the created dataset. Adjust the ratio of correct to incorrect samples in the training dataset to 0.5, and then execute the code below.

```bash
bash train_prober.sh
```

## Evaluation
Finally, you are able to evaluate the QA performance of our Probing-RAG with the following code! 
```bash
bash rag.sh
```

## Citation

