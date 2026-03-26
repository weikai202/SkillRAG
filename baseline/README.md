# Baseline

这个目录是一个完全隔离的 baseline 子系统。

约束：
- 所有新增代码都只放在 `baseline/`
- 数据下载到 `baseline/raw_data/`
- BM25 索引写到 `baseline/raw_data/sparse_index/`
- 结果写到 `baseline/results/`
- 不修改项目根目录下任何已有文件

## 功能

- 下载 5 个数据集：`hotpotqa`、`2wikimultihopqa`、`musique`、`nq`、`trivia`
- 用与原项目兼容的命名方式构建 BM25 索引：
  `baseline/raw_data/sparse_index/llama_index_bm25_model_{dataset}_2.json`
- 运行三个 BM25 baseline：
  - `FLARE`
  - `DRAGIN`
  - `Adaptive-RAG`

## 安装

```bash
pip install -r baseline/requirements.txt
```

## 下载数据

```bash
bash baseline/download.sh
```

或者：

```bash
python -m baseline.download_data --datasets all
```

## 构建 BM25 索引

```bash
bash baseline/make_index.sh
```

或者单个数据集：

```bash
python -m baseline.build_bm25_index --dataset_name hotpotqa
```

## 跑单个 baseline

```bash
python -m baseline.run_baseline \
  --method flare \
  --dataset_name hotpotqa \
  --model_id google/gemma-2-9b-it \
  --split dev \
  --limit 500
```

## 一次跑完 3 个 baseline x 5 个数据集

```bash
bash baseline/run_dev500.sh google/gemma-2-9b-it
```

按论文草稿里的 3 个模型一次性跑完：

```bash
bash baseline/run_paper_9.sh
```

对应模型为：
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `Qwen/Qwen3-8B`
- `google/gemma-2-9b-it`

跑完后生成论文表格所需的 baseline 汇总：

```bash
python -m baseline.summarize_paper_results
```

## 输出

- 明细：`baseline/results/<model_name>/<method>_<dataset>_<split>_<limit>.csv`
- 汇总：`baseline/results/<model_name>/<method>_<dataset>_<split>_<limit>.summary.json`
- 论文汇总：`baseline/results/paper_baseline_summary.csv`

## 说明

这里的三种方法都是为当前项目做的 BM25 兼容实现：
- `FLARE`：低置信草稿触发 query reformulation + 检索
- `DRAGIN`：高熵草稿触发检索和迭代修正
- `Adaptive-RAG`：先路由到 no-retrieval / single-hop / multi-hop，再执行对应策略

目标是先满足“使用当前 index 格式跑 baseline”的工程需求。

注意：这里不是对 FLARE / DRAGIN / Adaptive-RAG 官方仓库的逐行复现，而是 BM25-compatible 实现。若论文定稿阶段需要主张“严格复现原始 baseline”，还需要继续在 `baseline/` 内替换 router、query 生成器、stopping 逻辑，或直接对接官方实现。
