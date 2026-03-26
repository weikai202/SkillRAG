#!/usr/bin/env bash
set -euo pipefail

python -m baseline.build_bm25_index --dataset_name hotpotqa
python -m baseline.build_bm25_index --dataset_name nq
python -m baseline.build_bm25_index --dataset_name trivia
python -m baseline.build_bm25_index --dataset_name 2wikimultihopqa
python -m baseline.build_bm25_index --dataset_name musique
