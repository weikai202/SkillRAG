#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="probing"
PY_VER="3.10"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Please install Miniconda/Anaconda first."
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] conda env '${ENV_NAME}' already exists. Skip create."
else
  echo "[INFO] Creating conda env '${ENV_NAME}' with python=${PY_VER}..."
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

echo "[INFO] Activating env '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

echo "[INFO] Installing python packages..."
python -m pip install --upgrade pip
python -m pip install git+https://github.com/jbloomAus/SAELens
python -m pip install torch
python -m pip install einops
python -m pip install datasets
python -m pip install tqdm
python -m pip install wandb
python -m pip install faiss-cpu
python -m pip install ir-datasets
python -m pip install -U sentence-transformers
python -m pip install nltk
python -m pip install llama-index
python -m pip install ftfy
python -m pip install llama-index-retrievers-bm25
python -m pip install base58
python -m pip install spacy

echo "[INFO] Downloading spaCy model en_core_web_sm..."
python -m spacy download en_core_web_sm

echo "[INFO] Running environment sanity checks..."
python - <<'PY'
import importlib
import sys

pkgs = [
    "torch",
    "einops",
    "datasets",
    "tqdm",
    "wandb",
    "faiss",
    "ir_datasets",
    "sentence_transformers",
    "nltk",
    "llama_index",
    "ftfy",
    "base58",
    "spacy",
]

missing = []
for p in pkgs:
    try:
        importlib.import_module(p)
    except Exception:
        missing.append(p)

if missing:
    print("[ERROR] Missing imports:", ", ".join(missing))
    sys.exit(1)

import torch
import spacy

print("[OK] Python:", sys.version.split()[0])
print("[OK] Torch:", torch.__version__)
print("[OK] CUDA available:", torch.cuda.is_available())

try:
    nlp = spacy.load("en_core_web_sm")
    print("[OK] spaCy model loaded:", nlp.meta.get("name", "en_core_web_sm"))
except Exception as e:
    print("[ERROR] Failed to load spaCy model en_core_web_sm:", e)
    sys.exit(1)

print("[DONE] Environment is ready.")
PY

echo "[DONE] Setup completed for env '${ENV_NAME}'."
