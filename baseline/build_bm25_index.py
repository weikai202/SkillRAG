from __future__ import annotations

import argparse

from baseline.common import RAW_DATA_ROOT
from baseline.data import DATASETS
from baseline.indexing import build_sparse_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build isolated BM25 indexes inside baseline/raw_data/sparse_index.")
    parser.add_argument("--dataset_name", required=True, choices=DATASETS)
    args = parser.parse_args()

    output_path = build_sparse_index(args.dataset_name, root=RAW_DATA_ROOT)
    print(output_path)


if __name__ == "__main__":
    main()
