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

