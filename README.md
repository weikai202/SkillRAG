# SkillRAG

## Abstract

## Installation
```bash
conda create -n probing python=3.10
conda activate probing

pip install git+https://github.com/jbloomAus/SAELens
pip install torch
pip install einops
pip install datasets
pip install tqdm
pip install wandb
pip install faiss-cpu
pip install ir-datasets
pip install -U sentence-transformers
pip install nltk
pip install llama-index
pip install ftfy
pip install llama-index-retrievers-bm25
pip install base58
pip install spacy

python -m spacy download en_core_web_sm
```
## Datasets
You can download datasets as follows:
```bash
bash download/download.sh
bash download/raw_data.sh
```


## Whole Pipeline
Change the modelID for different models.
---
python run_pipeline.py --config configs/gemma2_9b.yaml
---









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

