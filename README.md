# SkillRAG

## Abstract
Retrieval-Augmented Generation (RAG) grounds large language models in external knowledge by querying document collections at inference time. While adaptive retrieval has improved efficiency, existing approaches treat post-retrieval failure as a retry signal rather than a diagnostic one, leaving the structural causes of query-evidence misalignment unaddressed. Most persistent retrieval failures stem not from missing evidence but from a mismatch between the query and the evidence space. We propose \textbf{Skill-RAG}, a failure-aware RAG framework that couples a lightweight hidden-state prober with a prompt-based skill router. The prober gates retrieval at two pipeline stages. On detecting a failure, the skill router diagnoses the cause and selects one of four corrective skills: query rewriting, question decomposition, evidence focusing, or an exit skill for truly irreducible cases. Across open-domain QA and complex reasoning benchmarks, Skill-RAG improves accuracy on hard cases that persist after multi-turn retrieval, with the largest gains on out-of-distribution datasets. Representation-space analyses show that the four skills occupy structured, separable regions of the failure state space, confirming that query-evidence misalignment is typed rather than monolithic.

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

## Project Structure
- `configs/*.yaml`: experiment presets. Each file defines the model, index datasets, retrieval mode, prober training layers, and evaluation methods.
- `run_pipeline.py`: end-to-end orchestrator for index building, prober data construction, balancing, prober training, checkpoint checks, evaluation, and report writing.
- `make_indexer.py`: builds sparse BM25 indexes or dense FAISS indexes from `raw_data/*`.
- `exp_rag.py`: main RAG runner. It supports `none`, `simple`, `probing`, and `skillrag`, and writes both per-sample traces and aggregate metrics.
- `balance_train_dataset.py`: balances generated prober training data by the `acc` label.
- `train.py`: trains hidden-state probers for the configured model layers.
- `check_prober_ckpt.py`: verifies that expected prober checkpoints exist before evaluation.
- `collect_build_metrics.py`: collects generated train/dev metrics for the pipeline report.
- `prompts.py`: stores QA, CoT, retrieval, diagnosis, routing, and skill prompts.
- `utils.py`: shared prober, retrieval, preprocessing, and evaluation utilities.
- `baseline/`: baseline implementations and saved baseline result summaries.
- `metrics/`: EM/F1 metric implementations.

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

Experiment stages in `run_pipeline.py`:
1. Build sparse indexes with `make_indexer.py` for `index_datasets`.
2. Generate prober construction data by running `exp_rag.py` with `simple` and `none` on train/dev splits for `build_dataset.datasets`.
3. Merge `simple` and `none` outputs inside `exp_rag.py` to produce `all_train_in3_` and `all_zeroshot_test_500` files.
4. Balance the training split with `balance_train_dataset.py`.
5. Train layer-wise hidden-state probers with `train.py`.
6. Check expected checkpoints with `check_prober_ckpt.py`.
7. Evaluate configured methods on `evaluate.datasets`.
8. Save a YAML run report under `reports/`.

Default experiment settings in `configs/gemma2_9b.yaml`:
- Model: `google/gemma-2-9b-it`
- Retrieval: sparse BM25 with CoT prompting
- Prober method: `tokens_mean`
- Prober layers: `12, 16, 20, 24, 28, 32, 36, 40`
- Training epochs: `2`
- Train/dev construction size: `3000` train samples and `500` dev samples per build dataset
- Evaluation size: `500` dev samples per evaluation dataset
- SkillRAG max retrieval rounds: `3`
- Prober decision threshold: `0.0`

Prober checkpoint selection during evaluation:
- For `nq/hotpotqa/trivia`: pass `--prober_train_dataset` as the same dataset
- For `musique/2wikimultihopqa`: no `--prober_train_dataset` argument (default fallback to `nq`)

Main outputs:
- Prober checkpoints: `ckpt/_3/<dataset_name>/...` and `pckpt/_3/<dataset_name>/...`
- Per-sample generation csv: `dataset/{2b|8b|9b}/retrieval_qa_*.csv`
- Aggregated eval metrics csv: `result/*.csv`
- Run report yaml: `reports/*.yaml`
- Appended metrics log: `result/metrics_log.yaml`

SkillRAG-specific traces:
- `initial_output`: the answer generated before corrective retrieval.
- `round_logs`: JSON text with each retrieval round's prober scores, selected skill, diagnosis, search query, retrieved evidence, and stopping decision.

