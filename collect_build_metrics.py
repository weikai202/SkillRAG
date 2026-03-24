import argparse
import glob
import os
from typing import Dict, List

import pandas as pd


MODEL_TO_DIR = {
    "google/gemma-2b": "2b",
    "meta-llama/Meta-Llama-3-8B-Instruct": "8b",
    "Qwen/Qwen3-8B": "8b",
    "google/gemma-2-9b-it": "9b",
}


def summarize_file(path: str, model_id: str, dataset_name: str, split: str, method: str) -> Dict:
    df = pd.read_csv(path)
    if len(df) == 0:
        return {
            "model_id": model_id,
            "dataset_name": dataset_name,
            "split": split,
            "retr_method": method,
            "count": 0,
            "acc": None,
            "em": None,
            "file": path,
        }
    return {
        "model_id": model_id,
        "dataset_name": dataset_name,
        "split": split,
        "retr_method": method,
        "count": int(len(df)),
        "acc": float(df["acc"].mean()) if "acc" in df.columns else None,
        "em": float(df["em"].mean()) if "em" in df.columns else None,
        "file": path,
    }


def collect_metrics(model_id: str, datasets: List[str], splits: List[str], methods: List[str]) -> List[Dict]:
    model_short = model_id.split("/")[-1]
    save_dir = MODEL_TO_DIR[model_id]
    out: List[Dict] = []
    for dataset_name in datasets:
        for split in splits:
            for method in methods:
                pattern = (
                    f"dataset/{save_dir}/retrieval_qa_{model_short}_{dataset_name}_{method}_{split}_*.csv"
                )
                matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
                if not matches:
                    out.append(
                        {
                            "model_id": model_id,
                            "dataset_name": dataset_name,
                            "split": split,
                            "retr_method": method,
                            "count": 0,
                            "acc": None,
                            "em": None,
                            "file": None,
                        }
                    )
                    continue
                out.append(summarize_file(matches[0], model_id, dataset_name, split, method))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, choices=list(MODEL_TO_DIR.keys()))
    parser.add_argument("--datasets", nargs="+", default=["trivia", "hotpotqa", "nq"])
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--methods", nargs="+", default=["simple", "none"])
    args = parser.parse_args()
    result = collect_metrics(args.model_id, args.datasets, args.splits, args.methods)
    print(result)


if __name__ == "__main__":
    main()

