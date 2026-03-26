from __future__ import annotations

import argparse

from baseline.common import RAW_DATA_ROOT, ensure_dir
from baseline.data import DATASETS, download_2wikimultihopqa, download_all, download_hotpotqa, download_musique, download_nq, download_trivia


def main() -> None:
    parser = argparse.ArgumentParser(description="Download baseline datasets into baseline/raw_data.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS + ["all"])
    args = parser.parse_args()

    ensure_dir(RAW_DATA_ROOT)
    selected = DATASETS if "all" in args.datasets else args.datasets
    for dataset_name in selected:
        if dataset_name == "nq":
            download_nq(RAW_DATA_ROOT)
        elif dataset_name == "trivia":
            download_trivia(RAW_DATA_ROOT)
        elif dataset_name == "hotpotqa":
            download_hotpotqa(RAW_DATA_ROOT)
        elif dataset_name == "2wikimultihopqa":
            download_2wikimultihopqa(RAW_DATA_ROOT)
        elif dataset_name == "musique":
            download_musique(RAW_DATA_ROOT)


if __name__ == "__main__":
    main()
